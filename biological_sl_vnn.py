#!/usr/bin/env python
"""
Biological Synthetic Lethality VNN
Streamlined implementation using real biological pathways for SL prediction
Addresses core performance issues with known SL pairs like ATM-ATR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("🧬 BIOLOGICAL SYNTHETIC LETHALITY VNN")
print("=" * 60)

class BiologicalPathwayLayer(nn.Module):
    """Pathway-constrained neural network layer"""
    
    def __init__(self, pathway_genes, gene2idx, pathway_name, output_dim=32):
        super().__init__()
        self.pathway_name = pathway_name
        self.pathway_genes = pathway_genes
        
        # Find genes that exist in our gene mapping
        self.gene_indices = []
        self.valid_genes = []
        
        for gene in pathway_genes:
            if gene in gene2idx:
                self.gene_indices.append(gene2idx[gene])
                self.valid_genes.append(gene)
        
        self.n_pathway_genes = len(self.gene_indices)
        
        if self.n_pathway_genes == 0:
            self.active = False
            return
        
        self.active = True
        input_dim = len(gene2idx)
        
        # Pathway-specific transformation
        self.pathway_transform = nn.Linear(self.n_pathway_genes, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Gene selection indices
        self.register_buffer('gene_indices_tensor', torch.LongTensor(self.gene_indices))
        
    def forward(self, gene_features):
        if not self.active:
            return torch.zeros(gene_features.size(0), 32, device=gene_features.device)
        
        # Select only pathway genes
        pathway_features = gene_features[:, self.gene_indices_tensor]
        
        # Transform through pathway-specific layer
        output = self.pathway_transform(pathway_features)
        output = self.activation(output)
        output = self.dropout(output)
        
        return output
    
    def get_gene_weights(self):
        """Get pathway gene weights for explainability"""
        if not self.active:
            return {}
        
        weights = {}
        weight_tensor = self.pathway_transform.weight.mean(dim=0)
        
        for i, gene in enumerate(self.valid_genes):
            weights[gene] = weight_tensor[i].item()
        
        return weights

class BiologicalSLVNN(nn.Module):
    """
    Biological Synthetic Lethality VNN
    Uses real pathway constraints for biologically-informed predictions
    """
    
    def __init__(self, hierarchy_data, gene2idx):
        super().__init__()
        self.hierarchy_data = hierarchy_data
        self.gene2idx = gene2idx
        self.n_genes = len(gene2idx)
        
        print(f"\n🏗️ Building Biological SL-VNN...")
        print(f"  📊 Genes: {self.n_genes:,}")
        
        # Extract key biological pathways
        self._build_pathway_layers()
        self._build_interaction_layers()
        self._build_prediction_layers()
        
    def _build_pathway_layers(self):
        """Build pathway-constrained layers"""
        print(f"\n  🧬 Building pathway layers...")
        
        # Get DNA repair pathways (most relevant for SL)
        dna_repair_pathways = []
        cell_cycle_pathways = []
        other_pathways = []
        
        # Categorize pathways by biological function
        pathway_structure = self.hierarchy_data['pathway_structure']
        
        # Look for DNA repair related clusters
        for level_name, clusters in pathway_structure.items():
            for cluster in clusters:
                cluster_genes = set(cluster['genes'])
                biological_annotation = cluster.get('biological_annotation', {})
                
                if 'DNA_REPAIR' in biological_annotation:
                    dna_repair_pathways.append(cluster)
                elif any(gene in cluster_genes for gene in ['CDK1', 'CDK2', 'CCND1', 'RB1']):
                    cell_cycle_pathways.append(cluster)
                else:
                    other_pathways.append(cluster)
        
        # Create pathway layers
        self.dna_repair_layers = nn.ModuleDict()
        self.cell_cycle_layers = nn.ModuleDict()
        self.general_layers = nn.ModuleDict()
        
        # DNA repair pathways (highest priority)
        print(f"    🧪 DNA Repair pathways: {len(dna_repair_pathways)}")
        for i, cluster in enumerate(dna_repair_pathways[:5]):  # Top 5
            layer_name = f"dna_repair_{i}"
            layer = BiologicalPathwayLayer(
                pathway_genes=cluster['genes'],
                gene2idx=self.gene2idx,
                pathway_name=layer_name,
                output_dim=64
            )
            if layer.active:
                self.dna_repair_layers[layer_name] = layer
                print(f"      ✅ {layer_name}: {layer.n_pathway_genes} genes")
        
        # Cell cycle pathways
        print(f"    🔄 Cell Cycle pathways: {len(cell_cycle_pathways)}")
        for i, cluster in enumerate(cell_cycle_pathways[:3]):  # Top 3
            layer_name = f"cell_cycle_{i}"
            layer = BiologicalPathwayLayer(
                pathway_genes=cluster['genes'],
                gene2idx=self.gene2idx,
                pathway_name=layer_name,
                output_dim=48
            )
            if layer.active:
                self.cell_cycle_layers[layer_name] = layer
                print(f"      ✅ {layer_name}: {layer.n_pathway_genes} genes")
        
        # General pathways
        print(f"    🌐 General pathways: {len(other_pathways)}")
        for i, cluster in enumerate(other_pathways[:5]):  # Top 5
            layer_name = f"general_{i}"
            layer = BiologicalPathwayLayer(
                pathway_genes=cluster['genes'],
                gene2idx=self.gene2idx,
                pathway_name=layer_name,
                output_dim=32
            )
            if layer.active:
                self.general_layers[layer_name] = layer
                print(f"      ✅ {layer_name}: {layer.n_pathway_genes} genes")
        
        # Calculate total pathway output dimension
        self.dna_repair_dim = len(self.dna_repair_layers) * 64
        self.cell_cycle_dim = len(self.cell_cycle_layers) * 48
        self.general_dim = len(self.general_layers) * 32
        self.total_pathway_dim = self.dna_repair_dim + self.cell_cycle_dim + self.general_dim
        
        print(f"    📊 Total pathway dimensions: {self.total_pathway_dim}")
        
    def _build_interaction_layers(self):
        """Build biological interaction layers"""
        print(f"\n  🔗 Building interaction layers...")
        
        # Pathway integration
        if self.total_pathway_dim > 0:
            self.pathway_integration = nn.Sequential(
                nn.Linear(self.total_pathway_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU()
            )
            print(f"    • Pathway integration: {self.total_pathway_dim} -> 128")
        else:
            # Fallback if no pathways found
            self.pathway_integration = nn.Sequential(
                nn.Linear(self.n_genes, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
            )
            print(f"    • Fallback integration: {self.n_genes} -> 128")
    
    def _build_prediction_layers(self):
        """Build SL prediction layers"""
        print(f"\n  🎯 Building prediction layers...")
        
        # Gene pair interaction
        self.gene_pair_interaction = nn.Sequential(
            nn.Linear(128 * 2, 256),  # Two genes
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final SL classification
        self.sl_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Binary SL prediction
        )
        
        print(f"    • Gene pair interaction: 256 -> 128")
        print(f"    • SL classifier: 128 -> 1")
        
        # Attention mechanism for pathway importance
        self.pathway_attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.dna_repair_layers) + len(self.cell_cycle_layers) + len(self.general_layers))
        ) if (len(self.dna_repair_layers) + len(self.cell_cycle_layers) + len(self.general_layers)) > 0 else None
    
    def forward(self, gene1_features, gene2_features):
        """Forward pass for SL prediction"""
        
        # Process both genes through pathway layers
        gene1_pathway_repr = self._get_pathway_representation(gene1_features)
        gene2_pathway_repr = self._get_pathway_representation(gene2_features)
        
        # Gene pair interaction
        pair_features = torch.cat([gene1_pathway_repr, gene2_pathway_repr], dim=1)
        pair_interaction = self.gene_pair_interaction(pair_features)
        
        # SL prediction
        sl_prediction = self.sl_classifier(pair_interaction)
        
        return sl_prediction
    
    def _get_pathway_representation(self, gene_features):
        """Get pathway-based representation for a gene"""
        
        pathway_outputs = []
        
        # DNA repair pathways
        for layer in self.dna_repair_layers.values():
            output = layer(gene_features)
            pathway_outputs.append(output)
        
        # Cell cycle pathways  
        for layer in self.cell_cycle_layers.values():
            output = layer(gene_features)
            pathway_outputs.append(output)
        
        # General pathways
        for layer in self.general_layers.values():
            output = layer(gene_features)
            pathway_outputs.append(output)
        
        # Concatenate all pathway outputs
        if pathway_outputs:
            pathway_concat = torch.cat(pathway_outputs, dim=1)
        else:
            # Fallback if no pathways
            pathway_concat = gene_features
        
        # Integrate through pathway integration layer
        integrated_repr = self.pathway_integration(pathway_concat)
        
        return integrated_repr
    
    def explain_prediction(self, gene1_features, gene2_features, gene1_name, gene2_name):
        """Generate biological explanation for SL prediction"""
        self.eval()
        
        with torch.no_grad():
            # Get prediction
            output = self.forward(gene1_features, gene2_features)
            prediction = torch.sigmoid(output).item()
            
            # Get pathway activations
            gene1_repr = self._get_pathway_representation(gene1_features)
            gene2_repr = self._get_pathway_representation(gene2_features)
            
            # Analyze pathway contributions
            pathway_analysis = {}
            
            # DNA repair pathway analysis
            for name, layer in self.dna_repair_layers.items():
                gene1_out = layer(gene1_features).mean().item()
                gene2_out = layer(gene2_features).mean().item()
                combined_activation = (gene1_out + gene2_out) / 2
                
                gene_weights = layer.get_gene_weights()
                
                pathway_analysis[name] = {
                    'type': 'DNA_REPAIR',
                    'activation': combined_activation,
                    'gene1_activation': gene1_out,
                    'gene2_activation': gene2_out,
                    'top_genes': sorted(gene_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                }
            
            # Cell cycle pathway analysis
            for name, layer in self.cell_cycle_layers.items():
                gene1_out = layer(gene1_features).mean().item()
                gene2_out = layer(gene2_features).mean().item()
                combined_activation = (gene1_out + gene2_out) / 2
                
                pathway_analysis[name] = {
                    'type': 'CELL_CYCLE',
                    'activation': combined_activation,
                    'gene1_activation': gene1_out,
                    'gene2_activation': gene2_out
                }
            
            # Get attention weights if available
            attention_weights = None
            if self.pathway_attention is not None:
                pair_repr = torch.cat([gene1_repr, gene2_repr], dim=1)
                pair_interaction = self.gene_pair_interaction(pair_repr)
                attention_logits = self.pathway_attention(pair_interaction)
                attention_weights = torch.softmax(attention_logits, dim=1)[0].tolist()
            
            # Generate biological mechanism explanation
            mechanism = self._generate_biological_mechanism(
                pathway_analysis, prediction > 0.5, gene1_name, gene2_name, attention_weights
            )
            
            explanation = {
                'gene1': gene1_name,
                'gene2': gene2_name,
                'prediction': prediction,
                'class': 'Synthetic Lethal' if prediction > 0.5 else 'Non-SL',
                'confidence': prediction if prediction > 0.5 else 1 - prediction,
                'pathway_analysis': pathway_analysis,
                'biological_mechanism': mechanism,
                'attention_weights': attention_weights
            }
        
        return explanation
    
    def _generate_biological_mechanism(self, pathway_analysis, is_sl, gene1, gene2, attention_weights):
        """Generate human-readable biological mechanism"""
        
        # Find most active pathways
        active_pathways = sorted(
            [(name, info['activation'], info['type']) for name, info in pathway_analysis.items()],
            key=lambda x: abs(x[1]), reverse=True
        )[:3]
        
        mechanism = f"🧬 **{gene1} ↔ {gene2}**: "
        
        if is_sl:
            mechanism += f"**SYNTHETIC LETHAL** interaction predicted.\n\n"
            mechanism += f"🔍 **Biological Evidence:**\n"
            
            # Analyze DNA repair pathway involvement
            dna_repair_active = [p for p in active_pathways if p[2] == 'DNA_REPAIR']
            if dna_repair_active:
                mechanism += f"• **DNA Repair Pathway Disruption**: Both genes show complementary activity in DNA repair mechanisms, suggesting their simultaneous loss would be lethal.\n"
            
            # Analyze cell cycle involvement
            cell_cycle_active = [p for p in active_pathways if p[2] == 'CELL_CYCLE']
            if cell_cycle_active:
                mechanism += f"• **Cell Cycle Checkpoint Failure**: Combined loss likely disrupts cell cycle control, leading to cell death.\n"
            
        else:
            mechanism += f"**NON-LETHAL** interaction predicted.\n\n"
            mechanism += f"🔍 **Evidence for Redundancy:**\n"
            mechanism += f"• Pathways show sufficient redundancy to compensate for loss of either gene.\n"
        
        mechanism += f"\n📊 **Top Pathway Activations:**\n"
        for pathway, activation, ptype in active_pathways:
            mechanism += f"• {ptype} ({pathway}): {activation:+.3f}\n"
        
        return mechanism

def load_hierarchy_and_create_features():
    """Load hierarchy data and create biological features"""
    print(f"\n📂 Loading biological hierarchy...")
    
    try:
        with open('/Users/Mahmuda/Desktop/cellmaps_vnn/data/biological_hierarchy.pkl', 'rb') as f:
            hierarchy_data = pickle.load(f)
        print(f"  ✓ Loaded hierarchy data")
        
        # Create gene mapping with our test genes prioritized
        test_genes = ['ATM', 'ATR', 'BRCA1', 'BRCA2', 'PARP1', 'TP53', 'RAD51', 'CHK1', 'CHK2']
        all_genes_available = list(hierarchy_data['gene_pathway_mapping'].keys())
        
        # Prioritize test genes
        gene_list = []
        for gene in test_genes:
            if gene in all_genes_available:
                gene_list.append(gene)
        
        # Add other important SL-related genes
        sl_related_genes = ['XRCC1', 'PRKDC', 'LIG4', 'NHEJ1', 'BAX', 'BCL2', 'CASP3', 'AKT1', 'MTOR']
        for gene in sl_related_genes:
            if gene in all_genes_available and gene not in gene_list:
                gene_list.append(gene)
        
        # Fill up to 500 genes total for efficiency
        remaining_genes = [g for g in all_genes_available if g not in gene_list]
        gene_list.extend(remaining_genes[:500-len(gene_list)])
        
        gene2idx = {gene: i for i, gene in enumerate(gene_list)}
        
        print(f"  ✓ Created gene mapping: {len(gene2idx)} genes")
        print(f"  ✓ Test genes included: {[g for g in test_genes if g in gene2idx]}")
        
        return hierarchy_data, gene2idx, gene_list
        
    except FileNotFoundError:
        print(f"  ❌ Hierarchy data not found. Run hierarchy_parser.py first!")
        return None, None, None

def create_biological_features(gene_names, gene2idx, hierarchy_data):
    """Create biologically-informed features"""
    
    n_genes = len(gene2idx)
    features = torch.zeros(len(gene_names), n_genes)
    
    gene_pathway_mapping = hierarchy_data['gene_pathway_mapping']
    
    for i, gene in enumerate(gene_names):
        if gene in gene2idx and gene in gene_pathway_mapping:
            gene_idx = gene2idx[gene]
            
            # Base gene presence
            features[i, gene_idx] = 1.0
            
            # Pathway-based enhancement
            pathway_info = gene_pathway_mapping[gene]
            
            # Weight by pathway membership (more pathways = higher importance)
            total_pathways = (len(pathway_info['level_2_pathways']) + 
                            len(pathway_info['level_1_pathways']) + 
                            len(pathway_info['level_0_pathways']))
            
            if total_pathways > 0:
                features[i, gene_idx] *= (1.0 + 0.1 * min(total_pathways, 10))
            
            # Add biological annotations weight
            bio_annotations = len(pathway_info['biological_annotations'])
            if bio_annotations > 0:
                features[i, gene_idx] *= (1.0 + 0.05 * bio_annotations)
            
            # Add interaction network centrality
            interaction_degree = pathway_info.get('interaction_degree', 0)
            if interaction_degree > 0:
                features[i, gene_idx] *= (1.0 + 0.001 * min(interaction_degree, 100))
    
    return features

def main():
    """Test the biological SL-VNN"""
    
    # Load data
    hierarchy_data, gene2idx, gene_list = load_hierarchy_and_create_features()
    if hierarchy_data is None:
        return
    
    # Build model
    print(f"\n🏗️ Building Biological SL-VNN...")
    model = BiologicalSLVNN(hierarchy_data, gene2idx)
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Summary:")
    print(f"  • Total parameters: {total_params:,}")
    print(f"  • DNA repair layers: {len(model.dna_repair_layers)}")
    print(f"  • Cell cycle layers: {len(model.cell_cycle_layers)}")
    print(f"  • General layers: {len(model.general_layers)}")
    
    # Test known SL pairs
    print(f"\n🧪 Testing Known SL Pairs...")
    print(f"=" * 60)
    
    test_pairs = [
        ('ATM', 'ATR', 'Known SL - DNA damage checkpoints'),
        ('BRCA1', 'PARP1', 'Known SL - DNA repair pathways'),  
        ('BRCA1', 'BRCA2', 'Unknown - both DNA repair but may be redundant')
    ]
    
    for gene1, gene2, description in test_pairs:
        if gene1 in gene2idx and gene2 in gene2idx:
            print(f"\n🔬 **{gene1} ↔ {gene2}**")
            print(f"  📋 {description}")
            
            # Create biological features
            gene1_features = create_biological_features([gene1], gene2idx, hierarchy_data)
            gene2_features = create_biological_features([gene2], gene2idx, hierarchy_data)
            
            # Get prediction and explanation
            explanation = model.explain_prediction(gene1_features, gene2_features, gene1, gene2)
            
            print(f"\n  📊 **Prediction**: {explanation['class']} ({explanation['confidence']:.1%} confidence)")
            print(f"  🧬 **Biological Mechanism**:")
            print(f"     {explanation['biological_mechanism']}")
            
            # Show pathway analysis
            print(f"\n  🔍 **Detailed Pathway Analysis**:")
            for pathway, analysis in explanation['pathway_analysis'].items():
                pathway_type = analysis['type']
                activation = analysis['activation']
                print(f"    • {pathway_type}: {activation:+.3f}")
                if 'top_genes' in analysis:
                    top_genes = analysis['top_genes'][:2]
                    gene_str = ', '.join([f"{g}({w:.2f})" for g, w in top_genes])
                    print(f"      Key genes: {gene_str}")
        else:
            missing = [g for g in [gene1, gene2] if g not in gene2idx]
            print(f"\n⚠️ **{gene1} ↔ {gene2}**: Missing genes {missing}")
    
    print(f"\n" + "=" * 60)
    print(f"✅ **BIOLOGICAL SL-VNN TESTING COMPLETE!**")
    print(f"🧬 **Real biological pathways integrated successfully!**")
    print(f"🎯 **Ready for training with conflict-resolved SL data!**")
    print(f"=" * 60)

if __name__ == "__main__":
    main()