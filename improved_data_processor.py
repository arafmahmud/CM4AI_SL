#!/usr/bin/env python
"""
Improved Data Processor with Evidence-Based Conflict Resolution
Handles conflicting labels using biological knowledge and evidence quality
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

print("🔧 IMPROVED SYNTHETIC LETHALITY DATA PROCESSOR")
print("=" * 60)

class SLDataProcessor:
    def __init__(self):
        # Evidence type hierarchy (higher score = more reliable)
        self.evidence_hierarchy = {
            'CRISPR/CRISPRi': 10,                    # Most reliable
            'High Throughput': 8,
            'GenomeRNAi': 6,
            'Low Throughput': 5,
            'RNAi Screen': 4,
            'Drug Screen': 4,
            'Computational Prediction': 3,
            'Text Mining': 2,
            'Decipher': 2,
            'Synlethality': 1,
            'Unknown': 0                             # Least reliable
        }
        
        # Clinical validation - known synthetic lethal pairs
        self.known_sl_pairs = {
            ('ATM', 'ATR'): 'DNA damage checkpoint kinases',
            ('BRCA1', 'PARP1'): 'Classic SL pair - FDA approved therapy',
            ('BRCA2', 'PARP1'): 'Homologous recombination deficiency',
            ('TP53', 'MDM2'): 'p53 pathway regulation',
            ('CCNE1', 'FBXW7'): 'Cell cycle checkpoint',
            ('MYC', 'BRD4'): 'Transcriptional regulation',
            ('KRAS', 'STK11'): 'Metabolic dependency'
        }
        
        # Context-dependent pairs (may vary by cell line/genetic background)
        self.context_dependent = {
            ('BRCA1', 'PARP1'): 'BRCA1 mutation status dependent',
            ('BRCA2', 'PARP1'): 'BRCA2 mutation status dependent',
            ('TP53', 'CDKN1A'): 'p53 status dependent'
        }
    
    def load_data(self):
        """Load and preprocess SL and Non-SL datasets"""
        print("\n📊 Loading raw datasets...")
        
        self.sl_data = pd.read_csv('/Users/Mahmuda/Desktop/cellmaps_vnn/Synthetic_lethality_data/sl.txt', sep='\t')
        self.non_sl_data = pd.read_csv('/Users/Mahmuda/Desktop/cellmaps_vnn/Synthetic_lethality_data/non_sl.txt', sep='\t')
        
        print(f"  ✓ Raw SL entries: {len(self.sl_data):,}")
        print(f"  ✓ Raw Non-SL entries: {len(self.non_sl_data):,}")
        
        return self
    
    def extract_pairs_with_metadata(self):
        """Extract gene pairs with comprehensive metadata"""
        print("\n🧬 Extracting pairs with metadata...")
        
        def process_evidence_type(evidence_str):
            """Process complex evidence types"""
            if pd.isna(evidence_str):
                return ['Unknown']
            
            evidence_str = str(evidence_str)
            # Handle multiple evidence types
            if ';' in evidence_str:
                return [e.strip() for e in evidence_str.split(';')]
            elif '|' in evidence_str:
                return [e.strip() for e in evidence_str.split('|')]
            else:
                return [evidence_str.strip()]
        
        def extract_cell_lines(cell_line_str):
            """Extract individual cell lines"""
            if pd.isna(cell_line_str) or str(cell_line_str).lower() in ['unknown', 'tbd', 'nan']:
                return []
            
            cell_line_str = str(cell_line_str)
            return [line.strip() for line in cell_line_str.split(';') if line.strip()]
        
        def process_dataset(df, label):
            pairs = []
            for _, row in df.iterrows():
                if pd.notna(row['x_name']) and pd.notna(row['y_name']):
                    gene1, gene2 = str(row['x_name']).strip(), str(row['y_name']).strip()
                    pair = tuple(sorted([gene1, gene2]))
                    
                    evidence_types = process_evidence_type(row.get('rel_source'))
                    cell_lines = extract_cell_lines(row.get('cell_line'))
                    
                    # Calculate evidence score
                    evidence_score = max([self.evidence_hierarchy.get(ev, 0) for ev in evidence_types])
                    
                    pairs.append({
                        'pair': pair,
                        'gene1': pair[0],
                        'gene2': pair[1],
                        'label': label,
                        'evidence_types': evidence_types,
                        'evidence_score': evidence_score,
                        'cell_lines': cell_lines,
                        'num_cell_lines': len(cell_lines),
                        'pubmed_ids': str(row.get('pubmed_id', '')).split(';'),
                        'cancer_type': str(row.get('cancer', 'Unknown'))
                    })
            return pairs
        
        self.sl_pairs = process_dataset(self.sl_data, 'SL')
        self.non_sl_pairs = process_dataset(self.non_sl_data, 'Non-SL')
        
        print(f"  ✓ Processed SL pairs: {len(self.sl_pairs):,}")
        print(f"  ✓ Processed Non-SL pairs: {len(self.non_sl_pairs):,}")
        
        return self
    
    def resolve_conflicts(self):
        """Resolve conflicting labels using evidence-based approach"""
        print("\n⚖️  Resolving conflicts using evidence hierarchy...")
        
        # Group by gene pair
        pair_evidence = defaultdict(list)
        for pair_info in self.sl_pairs + self.non_sl_pairs:
            pair_evidence[pair_info['pair']].append(pair_info)
        
        # Find and resolve conflicts
        resolved_pairs = []
        conflict_stats = {
            'total_pairs': len(pair_evidence),
            'conflict_pairs': 0,
            'evidence_resolved': 0,
            'cell_line_resolved': 0,
            'clinical_override': 0,
            'marked_context_dependent': 0
        }
        
        for pair, entries in pair_evidence.items():
            labels = set([entry['label'] for entry in entries])
            
            if len(labels) == 1:
                # No conflict - use any entry (prefer highest evidence)
                best_entry = max(entries, key=lambda x: (x['evidence_score'], x['num_cell_lines']))
                resolved_pairs.append(best_entry)
            else:
                # Conflict detected
                conflict_stats['conflict_pairs'] += 1
                
                # Check if this is a clinically known pair
                if pair in self.known_sl_pairs:
                    # Clinical override - prioritize SL
                    sl_entries = [e for e in entries if e['label'] == 'SL']
                    if sl_entries:
                        best_entry = max(sl_entries, key=lambda x: (x['evidence_score'], x['num_cell_lines']))
                        best_entry['resolution_method'] = 'clinical_override'
                        best_entry['clinical_note'] = self.known_sl_pairs[pair]
                        resolved_pairs.append(best_entry)
                        conflict_stats['clinical_override'] += 1
                        continue
                
                # Check if context-dependent
                if pair in self.context_dependent:
                    # Mark as context-dependent and use majority vote weighted by evidence
                    sl_score = sum([e['evidence_score'] for e in entries if e['label'] == 'SL'])
                    non_sl_score = sum([e['evidence_score'] for e in entries if e['label'] == 'Non-SL'])
                    
                    if sl_score >= non_sl_score:
                        best_entry = max([e for e in entries if e['label'] == 'SL'], 
                                       key=lambda x: (x['evidence_score'], x['num_cell_lines']))
                    else:
                        best_entry = max([e for e in entries if e['label'] == 'Non-SL'], 
                                       key=lambda x: (x['evidence_score'], x['num_cell_lines']))
                    
                    best_entry['resolution_method'] = 'context_dependent'
                    best_entry['context_note'] = self.context_dependent[pair]
                    resolved_pairs.append(best_entry)
                    conflict_stats['marked_context_dependent'] += 1
                    continue
                
                # Evidence-based resolution
                sl_entries = [e for e in entries if e['label'] == 'SL']
                non_sl_entries = [e for e in entries if e['label'] == 'Non-SL']
                
                max_sl_evidence = max([e['evidence_score'] for e in sl_entries]) if sl_entries else 0
                max_non_sl_evidence = max([e['evidence_score'] for e in non_sl_entries]) if non_sl_entries else 0
                
                if max_sl_evidence > max_non_sl_evidence:
                    best_entry = max(sl_entries, key=lambda x: (x['evidence_score'], x['num_cell_lines']))
                    best_entry['resolution_method'] = 'evidence_quality'
                    resolved_pairs.append(best_entry)
                    conflict_stats['evidence_resolved'] += 1
                elif max_non_sl_evidence > max_sl_evidence:
                    best_entry = max(non_sl_entries, key=lambda x: (x['evidence_score'], x['num_cell_lines']))
                    best_entry['resolution_method'] = 'evidence_quality'
                    resolved_pairs.append(best_entry)
                    conflict_stats['evidence_resolved'] += 1
                else:
                    # Same evidence quality - use cell line breadth
                    sl_cell_lines = sum([e['num_cell_lines'] for e in sl_entries])
                    non_sl_cell_lines = sum([e['num_cell_lines'] for e in non_sl_entries])
                    
                    if sl_cell_lines >= non_sl_cell_lines:
                        best_entry = max(sl_entries, key=lambda x: x['num_cell_lines'])
                    else:
                        best_entry = max(non_sl_entries, key=lambda x: x['num_cell_lines'])
                    
                    best_entry['resolution_method'] = 'cell_line_breadth'
                    resolved_pairs.append(best_entry)
                    conflict_stats['cell_line_resolved'] += 1
        
        self.resolved_pairs = resolved_pairs
        
        # Print resolution statistics
        print(f"  📊 Conflict Resolution Results:")
        print(f"    • Total unique pairs: {conflict_stats['total_pairs']:,}")
        print(f"    • Conflicting pairs: {conflict_stats['conflict_pairs']:,}")
        print(f"    • Evidence-based resolution: {conflict_stats['evidence_resolved']:,}")
        print(f"    • Cell line-based resolution: {conflict_stats['cell_line_resolved']:,}")
        print(f"    • Clinical overrides: {conflict_stats['clinical_override']:,}")
        print(f"    • Context-dependent pairs: {conflict_stats['marked_context_dependent']:,}")
        
        return self
    
    def create_clean_dataset(self):
        """Create clean training dataset with resolved conflicts"""
        print("\n✨ Creating clean training dataset...")
        
        # Separate by final labels
        final_sl_pairs = [p for p in self.resolved_pairs if p['label'] == 'SL']
        final_non_sl_pairs = [p for p in self.resolved_pairs if p['label'] == 'Non-SL']
        
        print(f"  ✓ Final SL pairs: {len(final_sl_pairs):,}")
        print(f"  ✓ Final Non-SL pairs: {len(final_non_sl_pairs):,}")
        
        # Create gene mappings
        all_genes = set()
        for pair_info in self.resolved_pairs:
            all_genes.add(pair_info['gene1'])
            all_genes.add(pair_info['gene2'])
        
        self.gene2idx = {gene: idx for idx, gene in enumerate(sorted(all_genes))}
        self.idx2gene = {idx: gene for gene, idx in self.gene2idx.items()}
        
        print(f"  ✓ Unique genes: {len(self.gene2idx):,}")
        
        # Create training pairs
        self.train_pairs = []
        self.train_labels = []
        
        for pair_info in self.resolved_pairs:
            gene1, gene2 = pair_info['gene1'], pair_info['gene2']
            label = 1 if pair_info['label'] == 'SL' else 0
            
            self.train_pairs.append((gene1, gene2))
            self.train_labels.append(label)
        
        print(f"  ✓ Training pairs: {len(self.train_pairs):,}")
        print(f"  ✓ SL ratio: {np.mean(self.train_labels):.3f}")
        
        return self
    
    def validate_clinical_pairs(self):
        """Validate against known clinical SL pairs"""
        print("\n🏥 Clinical validation...")
        
        clinical_results = {}
        for pair, description in self.known_sl_pairs.items():
            # Check if pair exists in our dataset
            pair_entries = [p for p in self.resolved_pairs if p['pair'] == pair]
            
            if pair_entries:
                entry = pair_entries[0]
                predicted_label = entry['label']
                evidence_score = entry['evidence_score']
                resolution_method = entry.get('resolution_method', 'no_conflict')
                
                clinical_results[pair] = {
                    'found': True,
                    'predicted': predicted_label,
                    'evidence_score': evidence_score,
                    'resolution': resolution_method,
                    'description': description
                }
                
                status = "✅" if predicted_label == 'SL' else "❌"
                print(f"  {status} {pair[0]} ↔ {pair[1]}: {predicted_label} (score: {evidence_score}, {resolution_method})")
            else:
                clinical_results[pair] = {
                    'found': False,
                    'description': description
                }
                print(f"  ❓ {pair[0]} ↔ {pair[1]}: NOT FOUND in dataset")
        
        self.clinical_validation = clinical_results
        return self
    
    def get_processed_data(self):
        """Return processed data for model training"""
        return {
            'pairs': self.train_pairs,
            'labels': self.train_labels,
            'gene2idx': self.gene2idx,
            'idx2gene': self.idx2gene,
            'resolved_data': self.resolved_pairs,
            'clinical_validation': self.clinical_validation
        }
    
    def save_processed_data(self, output_path='/Users/Mahmuda/Desktop/cellmaps_vnn/data/processed_sl_data.pkl'):
        """Save processed data"""
        import pickle
        
        data = self.get_processed_data()
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n💾 Saved processed data to: {output_path}")
        return self

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Create processor and run full pipeline
    processor = SLDataProcessor()
    
    # Run processing pipeline
    processor.load_data()
    processor.extract_pairs_with_metadata()
    processor.resolve_conflicts()
    processor.create_clean_dataset()
    processor.validate_clinical_pairs()
    processor.save_processed_data()
    
    # Get final statistics
    data = processor.get_processed_data()
    
    print("\n" + "=" * 60)
    print("📊 FINAL PROCESSING SUMMARY")
    print("=" * 60)
    
    print(f"\n📈 Clean Dataset:")
    print(f"  • Training pairs: {len(data['pairs']):,}")
    print(f"  • Unique genes: {len(data['gene2idx']):,}")
    print(f"  • SL ratio: {np.mean(data['labels']):.3f}")
    
    print(f"\n🏥 Clinical Validation:")
    found_clinical = sum(1 for r in data['clinical_validation'].values() if r['found'])
    correct_clinical = sum(1 for r in data['clinical_validation'].values() 
                          if r['found'] and r['predicted'] == 'SL')
    print(f"  • Known SL pairs found: {found_clinical}/{len(processor.known_sl_pairs)}")
    print(f"  • Correctly labeled as SL: {correct_clinical}/{found_clinical}")
    
    print(f"\n✅ Ready for improved model training!")
    print(f"🔧 Next: Implement biological feature engineering and hierarchy integration!")