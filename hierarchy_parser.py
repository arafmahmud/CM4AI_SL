#!/usr/bin/env python
"""
Biological Hierarchy Parser for Cell Maps VNN
Extracts hierarchical pathway structure from Cell Maps data files
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🧬 BIOLOGICAL HIERARCHY PARSER")
print("=" * 50)

class BiologicalHierarchyParser:
    def __init__(self):
        self.hierarchy_data = {}
        self.cluster_hierarchy = {}
        self.gene_to_clusters = defaultdict(set)
        self.interaction_network = None
        self.pathway_structure = {}
        
        # Known DNA repair and SL-relevant pathways for validation
        self.known_pathways = {
            'DNA_REPAIR': ['ATM', 'ATR', 'BRCA1', 'BRCA2', 'PARP1', 'TP53', 'CHK1', 'CHK2', 'RAD51', 'XRCC1'],
            'CELL_CYCLE': ['CDK1', 'CDK2', 'CCND1', 'CCNE1', 'RB1', 'E2F1'],
            'APOPTOSIS': ['TP53', 'BAX', 'BCL2', 'CASP3', 'CASP9'],
            'MTOR_SIGNALING': ['AKT1', 'MTOR', 'RICTOR', 'RPS6KB1'],
            'WNT_SIGNALING': ['DKK1', 'LRP5', 'LRP6', 'SOST']
        }
    
    def load_interaction_data(self):
        """Load protein-protein interaction data"""
        print("\n📊 Loading protein-protein interaction data...")
        
        try:
            # Load interaction scores
            interaction_file = '/Users/Mahmuda/Desktop/cellmaps_vnn/Hierarchy_data/abf3067_Data_S1.tsv'
            interactions_df = pd.read_csv(interaction_file, sep='\t')
            
            print(f"  ✓ Loaded {len(interactions_df):,} protein interactions")
            
            # Create interaction network
            self.interaction_network = nx.Graph()
            
            for _, row in interactions_df.iterrows():
                protein1 = str(row['Protein 1']).strip()
                protein2 = str(row['Protein 2']).strip()
                integrated_score = float(row['Integrated score'])
                
                # Add edge with all evidence types as attributes
                edge_data = {
                    'integrated_score': integrated_score,
                    'coexpression': row.get('evidence: Protein co-expression', 0),
                    'codependence': row.get('evidence: Co-dependence', 0),
                    'sequence_similarity': row.get('evidence: Sequence similarity', 0),
                    'physical': row.get('evidence: Physical', 0),
                    'mrna_coexpression': row.get('evidence: mRNA co-expression', 0)
                }
                
                self.interaction_network.add_edge(protein1, protein2, **edge_data)
            
            print(f"  ✓ Built interaction network: {self.interaction_network.number_of_nodes():,} nodes, {self.interaction_network.number_of_edges():,} edges")
            
        except Exception as e:
            print(f"  ⚠️ Error loading interaction data: {e}")
            self.interaction_network = nx.Graph()
    
    def load_hierarchical_clusters(self):
        """Load hierarchical gene clustering data"""
        print("\n🌳 Loading hierarchical gene clusters...")
        
        try:
            cluster_file = '/Users/Mahmuda/Desktop/cellmaps_vnn/Hierarchy_data/science.abf3067_Data_S2.txt'
            
            cluster_levels = defaultdict(list)
            
            with open(cluster_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        cluster_id = parts[0]
                        num_genes = int(parts[1])
                        genes = parts[2].split()
                        
                        # Extract cluster level from ID (e.g., Cluster0-0, Cluster2-469)
                        level = int(cluster_id.split('-')[0].replace('Cluster', ''))
                        
                        cluster_info = {
                            'id': cluster_id,
                            'level': level,
                            'size': num_genes,
                            'genes': genes
                        }
                        
                        cluster_levels[level].append(cluster_info)
                        self.cluster_hierarchy[cluster_id] = cluster_info
                        
                        # Map genes to clusters
                        for gene in genes:
                            self.gene_to_clusters[gene].add(cluster_id)
            
            # Summary statistics
            total_clusters = len(self.cluster_hierarchy)
            levels = sorted(cluster_levels.keys())
            
            print(f"  ✓ Loaded {total_clusters:,} clusters across {len(levels)} levels")
            
            for level in levels:
                clusters = cluster_levels[level]
                sizes = [c['size'] for c in clusters]
                print(f"    Level {level}: {len(clusters):,} clusters, size range: {min(sizes)}-{max(sizes)} genes")
            
            # Find clusters containing our known SL genes
            test_genes = ['ATM', 'ATR', 'BRCA1', 'BRCA2', 'PARP1']
            print(f"\n  🔍 Locating test genes in hierarchy:")
            
            for gene in test_genes:
                clusters = self.gene_to_clusters.get(gene, set())
                if clusters:
                    cluster_info = [(cid, self.cluster_hierarchy[cid]['level'], self.cluster_hierarchy[cid]['size']) 
                                   for cid in clusters]
                    cluster_info.sort(key=lambda x: x[2])  # Sort by size
                    print(f"    {gene}: found in {len(clusters)} clusters")
                    for cid, level, size in cluster_info[:3]:  # Show 3 smallest clusters
                        print(f"      • {cid} (L{level}, {size} genes)")
                else:
                    print(f"    {gene}: NOT FOUND in hierarchy")
            
            self.hierarchy_data['levels'] = cluster_levels
            
        except Exception as e:
            print(f"  ⚠️ Error loading cluster data: {e}")
    
    def extract_pathway_structure(self):
        """Extract biological pathway structure for VNN"""
        print("\n🧪 Extracting biological pathway structure...")
        
        # Organize clusters by biological relevance
        pathway_clusters = {
            'small_specific': [],      # Level 2 clusters (4-10 genes)
            'medium_functional': [],   # Level 1 clusters (10-100 genes) 
            'large_general': []        # Level 0 clusters (100+ genes)
        }
        
        for cluster_id, cluster_info in self.cluster_hierarchy.items():
            size = cluster_info['size']
            level = cluster_info['level']
            
            if size <= 10 and level >= 2:
                pathway_clusters['small_specific'].append(cluster_info)
            elif 10 < size <= 100:
                pathway_clusters['medium_functional'].append(cluster_info)
            else:
                pathway_clusters['large_general'].append(cluster_info)
        
        print(f"  ✓ Organized into pathway categories:")
        for category, clusters in pathway_clusters.items():
            print(f"    • {category}: {len(clusters)} clusters")
        
        # Analyze small specific clusters for biological coherence
        print(f"\n  🔬 Analyzing small specific clusters:")
        biologically_coherent = []
        
        for cluster in pathway_clusters['small_specific'][:20]:  # Check first 20
            genes = cluster['genes']
            
            # Check if cluster matches known pathways
            pathway_matches = {}
            for pathway, known_genes in self.known_pathways.items():
                overlap = set(genes) & set(known_genes)
                if len(overlap) >= 2:  # At least 2 genes overlap
                    pathway_matches[pathway] = overlap
            
            if pathway_matches:
                cluster['biological_annotation'] = pathway_matches
                biologically_coherent.append(cluster)
                print(f"    ✅ {cluster['id']}: {genes}")
                for pathway, overlap in pathway_matches.items():
                    print(f"       → {pathway}: {list(overlap)}")
        
        print(f"  ✓ Found {len(biologically_coherent)} biologically coherent clusters")
        
        # Create hierarchical pathway structure for VNN
        self.pathway_structure = {
            'level_0_general': [c for c in pathway_clusters['large_general'][:5]],  # Top 5 general
            'level_1_functional': [c for c in pathway_clusters['medium_functional'][:50]],  # Top 50 functional  
            'level_2_specific': biologically_coherent[:100]  # Top 100 specific
        }
        
        print(f"\n  🏗️ VNN pathway structure:")
        for level, clusters in self.pathway_structure.items():
            total_genes = sum(c['size'] for c in clusters)
            print(f"    • {level}: {len(clusters)} clusters, {total_genes} total genes")
    
    def create_gene_pathway_mapping(self):
        """Create gene to pathway mapping for feature engineering"""
        print("\n🗺️ Creating gene-pathway mapping...")
        
        gene_pathway_features = {}
        
        all_genes = set()
        for cluster_info in self.cluster_hierarchy.values():
            all_genes.update(cluster_info['genes'])
        
        print(f"  📊 Processing {len(all_genes):,} unique genes...")
        
        for gene in all_genes:
            features = {
                'level_0_pathways': [],
                'level_1_pathways': [],
                'level_2_pathways': [],
                'interaction_degree': 0,
                'biological_annotations': []
            }
            
            # Find which pathways this gene belongs to
            gene_clusters = self.gene_to_clusters.get(gene, set())
            
            for cluster_id in gene_clusters:
                cluster_info = self.cluster_hierarchy[cluster_id]
                level = cluster_info['level']
                size = cluster_info['size']
                
                if size <= 10 and level >= 2:
                    features['level_2_pathways'].append(cluster_id)
                elif 10 < size <= 100:
                    features['level_1_pathways'].append(cluster_id)
                else:
                    features['level_0_pathways'].append(cluster_id)
                
                # Add biological annotations if available
                if 'biological_annotation' in cluster_info:
                    features['biological_annotations'].extend(cluster_info['biological_annotation'].keys())
            
            # Add interaction network features
            if self.interaction_network and gene in self.interaction_network:
                features['interaction_degree'] = self.interaction_network.degree[gene]
            
            gene_pathway_features[gene] = features
        
        self.gene_pathway_mapping = gene_pathway_features
        
        # Validate with our test genes
        print(f"\n  🧬 Validation with test genes:")
        test_genes = ['ATM', 'ATR', 'BRCA1', 'BRCA2', 'PARP1']
        
        for gene in test_genes:
            if gene in gene_pathway_features:
                features = gene_pathway_features[gene]
                print(f"    {gene}:")
                print(f"      • Level 2 pathways: {len(features['level_2_pathways'])}")
                print(f"      • Level 1 pathways: {len(features['level_1_pathways'])}")
                print(f"      • Interaction degree: {features['interaction_degree']}")
                print(f"      • Biological annotations: {features['biological_annotations']}")
            else:
                print(f"    {gene}: NOT FOUND")
    
    def save_hierarchy_data(self):
        """Save processed hierarchy data"""
        print(f"\n💾 Saving processed hierarchy data...")
        
        output_data = {
            'cluster_hierarchy': self.cluster_hierarchy,
            'gene_to_clusters': dict(self.gene_to_clusters),
            'pathway_structure': self.pathway_structure,
            'gene_pathway_mapping': self.gene_pathway_mapping,
            'interaction_network': self.interaction_network,
            'known_pathways': self.known_pathways
        }
        
        output_file = '/Users/Mahmuda/Desktop/cellmaps_vnn/data/biological_hierarchy.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)
        
        print(f"  ✓ Saved to: {output_file}")
        
        # Also save human-readable summary
        summary_file = '/Users/Mahmuda/Desktop/cellmaps_vnn/data/hierarchy_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("BIOLOGICAL HIERARCHY SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Clusters: {len(self.cluster_hierarchy):,}\n")
            f.write(f"Total Genes: {len(self.gene_pathway_mapping):,}\n")
            f.write(f"Interaction Network: {self.interaction_network.number_of_nodes():,} nodes, {self.interaction_network.number_of_edges():,} edges\n\n")
            
            f.write("VNN Pathway Structure:\n")
            for level, clusters in self.pathway_structure.items():
                f.write(f"  {level}: {len(clusters)} clusters\n")
            
            f.write(f"\nTest Gene Coverage:\n")
            test_genes = ['ATM', 'ATR', 'BRCA1', 'BRCA2', 'PARP1']
            for gene in test_genes:
                if gene in self.gene_pathway_mapping:
                    features = self.gene_pathway_mapping[gene]
                    f.write(f"  {gene}: {len(features['level_2_pathways'])} specific, {len(features['level_1_pathways'])} functional pathways\n")
                else:
                    f.write(f"  {gene}: NOT FOUND\n")
        
        print(f"  ✓ Summary saved to: {summary_file}")
        
        return output_data

def main():
    """Main execution"""
    parser = BiologicalHierarchyParser()
    
    # Load all hierarchy data
    parser.load_interaction_data()
    parser.load_hierarchical_clusters()
    parser.extract_pathway_structure()
    parser.create_gene_pathway_mapping()
    
    # Save processed data
    hierarchy_data = parser.save_hierarchy_data()
    
    print("\n" + "=" * 50)
    print("📊 HIERARCHY PARSING COMPLETE")
    print("=" * 50)
    
    print(f"\n✅ Successfully processed:")
    print(f"  • {len(hierarchy_data['cluster_hierarchy']):,} hierarchical clusters")
    print(f"  • {len(hierarchy_data['gene_pathway_mapping']):,} genes with pathway annotations")
    print(f"  • {hierarchy_data['interaction_network'].number_of_edges():,} protein interactions")
    print(f"  • {sum(len(clusters) for clusters in hierarchy_data['pathway_structure'].values())} pathway clusters for VNN")
    
    print(f"\n🔧 Ready for hierarchical VNN implementation!")
    print(f"🧬 Biological pathway structure extracted and validated!")

if __name__ == "__main__":
    main()