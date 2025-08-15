#!/usr/bin/env python
"""
Complete Training Pipeline for Biological SL-VNN
Implements actual training with hierarchy constraints and saves trained model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 BIOLOGICAL SL-VNN TRAINING PIPELINE")
print("=" * 60)

class SLDataset(Dataset):
    """Dataset for synthetic lethality pairs"""
    
    def __init__(self, pairs, labels, gene2idx, hierarchy_data):
        self.pairs = pairs
        self.labels = labels
        self.gene2idx = gene2idx
        self.hierarchy_data = hierarchy_data
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        gene1, gene2 = self.pairs[idx]
        label = self.labels[idx]
        
        # Create biological features for both genes
        gene1_features = self._create_biological_features([gene1])
        gene2_features = self._create_biological_features([gene2])
        
        return {
            'gene1_features': gene1_features.squeeze(0),
            'gene2_features': gene2_features.squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32),
            'gene1_name': gene1,
            'gene2_name': gene2
        }
    
    def _create_biological_features(self, gene_names):
        """Create biologically-informed features"""
        n_genes = len(self.gene2idx)
        features = torch.zeros(len(gene_names), n_genes)
        
        gene_pathway_mapping = self.hierarchy_data['gene_pathway_mapping']
        
        for i, gene in enumerate(gene_names):
            if gene in self.gene2idx and gene in gene_pathway_mapping:
                gene_idx = self.gene2idx[gene]
                
                # Base gene presence
                features[i, gene_idx] = 1.0
                
                # Pathway-based enhancement
                pathway_info = gene_pathway_mapping[gene]
                
                # Weight by pathway membership
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

class BiologicalSLVNNTrainer:
    """Complete training pipeline for biological SL-VNN"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': []
        }
        
    def create_loss_function(self):
        """Create loss function with pathway regularization"""
        
        def pathway_regularized_loss(outputs, targets, model, alpha=0.01):
            """Loss with biological pathway regularization"""
            
            # Standard binary cross-entropy
            bce_loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), targets)
            
            # Pathway regularization - encourage meaningful pathway weights
            pathway_reg = 0.0
            pathway_count = 0
            
            # DNA repair pathway regularization (most important for SL)
            for name, layer in model.dna_repair_layers.items():
                if hasattr(layer, 'pathway_transform'):
                    weights = layer.pathway_transform.weight
                    # Encourage sparse but strong connections
                    l1_reg = torch.sum(torch.abs(weights))
                    l2_reg = torch.sum(weights ** 2)
                    pathway_reg += l1_reg + 0.1 * l2_reg
                    pathway_count += 1
            
            # Cell cycle pathway regularization
            for name, layer in model.cell_cycle_layers.items():
                if hasattr(layer, 'pathway_transform'):
                    weights = layer.pathway_transform.weight
                    l1_reg = torch.sum(torch.abs(weights))
                    pathway_reg += 0.5 * l1_reg  # Less weight than DNA repair
                    pathway_count += 1
            
            # Normalize by number of pathways
            if pathway_count > 0:
                pathway_reg = pathway_reg / pathway_count
            
            total_loss = bce_loss + alpha * pathway_reg
            
            return total_loss, bce_loss, pathway_reg
        
        return pathway_regularized_loss
    
    def train_epoch(self, train_loader, optimizer, loss_fn):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_reg = 0.0
        all_outputs = []
        all_targets = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            gene1_features = batch['gene1_features'].to(self.device)
            gene2_features = batch['gene2_features'].to(self.device)
            targets = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(gene1_features, gene2_features)
            
            # Compute loss with pathway regularization
            loss, bce_loss, reg_loss = loss_fn(outputs, targets, self.model)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_bce += bce_loss.item()
            total_reg += reg_loss.item()
            
            # Store predictions for AUC calculation
            all_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'BCE': f'{bce_loss.item():.4f}',
                'Reg': f'{reg_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_bce = total_bce / len(train_loader)
        avg_reg = total_reg / len(train_loader)
        train_auc = roc_auc_score(all_targets, all_outputs)
        
        return avg_loss, avg_bce, avg_reg, train_auc
    
    def validate_epoch(self, val_loader, loss_fn):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                gene1_features = batch['gene1_features'].to(self.device)
                gene2_features = batch['gene2_features'].to(self.device)
                targets = batch['label'].to(self.device)
                
                outputs = self.model(gene1_features, gene2_features)
                loss, _, _ = loss_fn(outputs, targets, self.model)
                
                total_loss += loss.item()
                all_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        val_auc = roc_auc_score(all_targets, all_outputs)
        
        return avg_loss, val_auc
    
    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
        """Complete training pipeline"""
        
        print(f"\n🏋️ Starting training for {num_epochs} epochs...")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Validation samples: {len(val_loader.dataset)}")
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        loss_fn = self.create_loss_function()
        
        best_val_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, train_bce, train_reg, train_auc = self.train_epoch(train_loader, optimizer, loss_fn)
            
            # Validation
            val_loss, val_auc = self.validate_epoch(val_loader, loss_fn)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_auc'].append(train_auc)
            self.training_history['val_auc'].append(val_auc)
            
            print(f"📈 Train Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, Reg: {train_reg:.4f})")
            print(f"📈 Train AUC: {train_auc:.4f}")
            print(f"📉 Val Loss: {val_loss:.4f}")
            print(f"📉 Val AUC: {val_auc:.4f}")
            print(f"🎯 LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping and model saving
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'training_history': self.training_history
                }, '/Users/Mahmuda/Desktop/cellmaps_vnn/trained_biological_sl_vnn.pth')
                
                print(f"💾 New best model saved! AUC: {val_auc:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= 10:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                print(f"🏆 Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch+1}")
                break
        
        print(f"\n✅ Training completed!")
        print(f"🏆 Best validation AUC: {best_val_auc:.4f}")
        
        return best_val_auc
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # AUC plot
        ax2.plot(epochs, self.training_history['train_auc'], 'b-', label='Training AUC')
        ax2.plot(epochs, self.training_history['val_auc'], 'r-', label='Validation AUC')
        ax2.set_title('Training and Validation AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history saved to: {save_path}")
        
        plt.show()

def load_data():
    """Load processed data and hierarchy"""
    print("📂 Loading processed data...")
    
    # Load processed SL data
    with open('/Users/Mahmuda/Desktop/cellmaps_vnn/data/processed_sl_data.pkl', 'rb') as f:
        sl_data = pickle.load(f)
    
    # Load biological hierarchy
    with open('/Users/Mahmuda/Desktop/cellmaps_vnn/data/biological_hierarchy.pkl', 'rb') as f:
        hierarchy_data = pickle.load(f)
    
    print(f"  ✅ SL pairs: {len(sl_data['pairs']):,}")
    print(f"  ✅ Unique genes: {len(sl_data['gene2idx']):,}")
    print(f"  ✅ SL ratio: {np.mean(sl_data['labels']):.3f}")
    
    return sl_data, hierarchy_data

def create_gene_mapping(sl_data, hierarchy_data):
    """Create consistent gene mapping"""
    print("🧬 Creating consistent gene mapping...")
    
    # Get genes from both datasets
    sl_genes = set(sl_data['gene2idx'].keys())
    hierarchy_genes = set(hierarchy_data['gene_pathway_mapping'].keys())
    
    # Find intersection
    common_genes = sl_genes & hierarchy_genes
    print(f"  📊 SL genes: {len(sl_genes):,}")
    print(f"  📊 Hierarchy genes: {len(hierarchy_genes):,}")
    print(f"  📊 Common genes: {len(common_genes):,}")
    
    # Prioritize test genes and DNA repair genes
    priority_genes = ['ATM', 'ATR', 'BRCA1', 'BRCA2', 'PARP1', 'TP53', 'CHK1', 'CHK2', 'RAD51', 'XRCC1']
    gene_list = []
    
    # Add priority genes first
    for gene in priority_genes:
        if gene in common_genes:
            gene_list.append(gene)
    
    # Add remaining common genes (up to 1000 total for efficiency)
    remaining_genes = [g for g in common_genes if g not in gene_list]
    gene_list.extend(remaining_genes[:1000-len(gene_list)])
    
    # Create mapping
    gene2idx = {gene: i for i, gene in enumerate(gene_list)}
    
    print(f"  ✅ Final gene mapping: {len(gene2idx):,} genes")
    print(f"  ✅ Priority genes included: {[g for g in priority_genes if g in gene2idx]}")
    
    return gene2idx, gene_list

def filter_pairs_by_genes(pairs, labels, gene2idx):
    """Filter pairs to only include genes in our mapping"""
    filtered_pairs = []
    filtered_labels = []
    
    for (gene1, gene2), label in zip(pairs, labels):
        if gene1 in gene2idx and gene2 in gene2idx:
            filtered_pairs.append((gene1, gene2))
            filtered_labels.append(label)
    
    print(f"  📊 Original pairs: {len(pairs):,}")
    print(f"  📊 Filtered pairs: {len(filtered_pairs):,}")
    print(f"  📊 Filtered SL ratio: {np.mean(filtered_labels):.3f}")
    
    return filtered_pairs, filtered_labels

def main():
    """Main training pipeline"""
    
    # Load data
    sl_data, hierarchy_data = load_data()
    
    # Create consistent gene mapping
    gene2idx, gene_list = create_gene_mapping(sl_data, hierarchy_data)
    
    # Filter pairs to common genes
    filtered_pairs, filtered_labels = filter_pairs_by_genes(
        sl_data['pairs'], sl_data['labels'], gene2idx
    )
    
    # Split into train/validation
    print("\n🔄 Creating train/validation split...")
    train_pairs, val_pairs, train_labels, val_labels = train_test_split(
        filtered_pairs, filtered_labels, test_size=0.2, random_state=42, stratify=filtered_labels
    )
    
    print(f"  📊 Training pairs: {len(train_pairs):,}")
    print(f"  📊 Validation pairs: {len(val_pairs):,}")
    print(f"  📊 Train SL ratio: {np.mean(train_labels):.3f}")
    print(f"  📊 Val SL ratio: {np.mean(val_labels):.3f}")
    
    # Create datasets
    train_dataset = SLDataset(train_pairs, train_labels, gene2idx, hierarchy_data)
    val_dataset = SLDataset(val_pairs, val_labels, gene2idx, hierarchy_data)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Import and create model
    from biological_sl_vnn import BiologicalSLVNN
    
    print(f"\n🏗️ Building Biological SL-VNN...")
    model = BiologicalSLVNN(hierarchy_data, gene2idx)
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Summary:")
    print(f"  • Total parameters: {total_params:,}")
    print(f"  • Trainable parameters: {trainable_params:,}")
    print(f"  • DNA repair layers: {len(model.dna_repair_layers)}")
    print(f"  • Cell cycle layers: {len(model.cell_cycle_layers)}")
    print(f"  • General layers: {len(model.general_layers)}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")
    
    # Create trainer and train
    trainer = BiologicalSLVNNTrainer(model, device)
    
    # Train the model
    best_auc = trainer.train(train_loader, val_loader, num_epochs=100, learning_rate=0.001)
    
    # Plot training history
    trainer.plot_training_history('/Users/Mahmuda/Desktop/cellmaps_vnn/training_history.png')
    
    # Save final results
    results = {
        'best_validation_auc': best_auc,
        'gene2idx': gene2idx,
        'gene_list': gene_list,
        'training_history': trainer.training_history,
        'model_architecture_summary': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'dna_repair_layers': len(model.dna_repair_layers),
            'cell_cycle_layers': len(model.cell_cycle_layers),
            'general_layers': len(model.general_layers)
        }
    }
    
    with open('/Users/Mahmuda/Desktop/cellmaps_vnn/training_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n" + "=" * 60)
    print(f"🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"🏆 Best Validation AUC: {best_auc:.4f}")
    print(f"💾 Trained model saved to: trained_biological_sl_vnn.pth")
    print(f"📊 Training history saved to: training_history.png")
    print(f"📋 Results saved to: training_results.pkl")
    print(f"=" * 60)

if __name__ == "__main__":
    main()