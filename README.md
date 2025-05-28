# SHARP-Distill: A 68× Faster Recommender System with Hypergraph Neural Networks and Language Models

[![ICML 2025](https://img.shields.io/badge/ICML-2025-blue.svg)](https://icml.cc/Conferences/2025)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

**SHARP-Distill** (**Speedy Hypergraph And Review-based Personalised Distillation**) is a novel knowledge distillation framework that combines Hypergraph Neural Networks (HGNNs) with language models to enhance recommendation quality while achieving **68× faster inference** than HGNN and **40× faster** than LightGCN. Accepted at **ICML 2025**.

### Key Features

- 🚀 **68× faster inference** than HGNN, **40× faster** than LightGCN
- 🔗 **Hypergraph Neural Networks** for capturing high-order user-item relationships  
- 🧠 **Advanced Knowledge Distillation** using teacher-student framework
- 📊 **Multi-modal fusion** of structural (HGNN) and semantic (DeBERTa) embeddings
- ⚡ **CompactGCN**: Lightweight single-layer student architecture
- 🎯 **Contrastive learning** for structural and positional knowledge transfer
- 📈 **State-of-the-art accuracy** with production-ready efficiency

## 🏗️ Architecture

SHARP-Distill implements a teacher-student framework with three core components:

### 🎓 Teacher Model
- **Dual HGNNs**: Capture high-order user-item interactions using hypergraph Laplacian
- **DeBERTa**: Extract semantic features from textual reviews with disentangled attention
- **Cross-modal alignment**: InfoNCE contrastive learning between structural and textual embeddings
- **Multi-layer MLP**: Fuse HGNN and DeBERTa embeddings for rating predictions

### 🎯 Student Model  
- **CompactGCN**: Single-layer GCN without non-linear activations for efficiency
- **Embedding interpolation**: Blend student and teacher embeddings during training
- **Lightweight MLP**: Fast inference with preserved high-order knowledge

### 🔄 Knowledge Transfer
- **Soft label distillation**: Temperature-scaled probability distributions
- **Contrastive alignment**: Structural and positional similarity preservation
- **Hypergraph positional encoding**: Transfer topological knowledge to student

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SHARP-Distill.git
cd SHARP-Distill

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
networkx>=2.6
scikit-learn>=1.0.0
transformers>=4.15.0
```

### Basic Usage

```python
from sharp_distill import main

# Run the complete SHARP-Distill pipeline
if __name__ == "__main__":
    main()
```

### Custom Dataset

```python
# Prepare your data in CSV format with columns: user_id, item_id, review
interactions_path = 'path/to/your/dataset.csv'

# Load and split data
train_data, validation_data, test_data = load_and_split_data(interactions_path)

# Create hypergraph incidence matrices
user_hypergraph, item_hypergraph, user_to_index, item_to_index = create_hypergraphs_and_incidence_matrices(train_data)

# Initialize models
lightgcn_model, hypergraph_model, mlp_model = initialize_models(num_nodes, user_hypergraph.shape[1])

# Train with knowledge distillation
mlp_model, hypergraph_model = train_models(
    mlp_model, hypergraph_model, 
    user_hypergraph, item_hypergraph
)
```

## 📊 Performance

### Speed Comparison

| Model | Inference Time (ms) | Speedup | Parameters |
|-------|-------------------|---------|------------|
| HGNN + DeBERTa (Teacher) | 668.23 | 1× | 145M |
| LightGCN | 395.45 | 1.7× | 1.5M |
| **SHARP-Distill** | **9.77** | **68×** | **0.5M** |

### Accuracy Metrics (Amazon CDs Dataset)

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 |
|-------|-------------|-----------|---------|-------------|
| LightGCN | 11.67 | 10.14 | 9.75 | 42.59 |
| HCCF | 13.96 | 12.05 | 11.70 | 49.72 |
| HGAtt | 13.21 | 12.47 | 12.24 | 48.55 |
| **SHARP-Distill** | **13.75** | **13.06** | **12.17** | **54.42** |

### Performance Retention

SHARP-Distill maintains **94.1% average performance retention** compared to the teacher model while achieving dramatic speedups across all datasets.

## 🔧 Configuration

### Model Parameters

```python
# Hypergraph Neural Network
hypergraph_config = {
    'input_dim': 768,        # DeBERTa embedding dimension
    'hidden_dim': 64,        # HGNN hidden dimension  
    'output_dim': 64,        # Final embedding dimension
    'T_max': 10.0,          # Maximum temperature
    'T_min': 1.0,           # Minimum temperature
    'total_epochs': 200      # Training epochs
}

# CompactGCN Student
compactgcn_config = {
    'input_dim': 64,         # Input embedding dimension
    'hidden_dim': 64,        # Hidden layer dimension
    'num_layers': 1,         # Single layer for efficiency
    'activation': None       # No non-linear activation
}

# Knowledge Distillation
distillation_config = {
    'temperature': 1.0,       # Soft label temperature
    'contrastive_temp': 0.1,  # Contrastive learning temperature
    'alpha': 0.5,            # Embedding vs positional similarity weight
    'gamma': 0.5,            # Interpolation coefficient
    'lambda_1': 1.0,         # KL divergence loss weight
    'lambda_2': 1.0,         # User contrastive loss weight  
    'lambda_3': 1.0          # Item contrastive loss weight
}
```

### Training Parameters

```python
training_config = {
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 200,
    'reg_lambda': 0.001,
    'gamma': 0.5  # Interpolation factor
}
```

## 📁 Project Structure

```
SHARP-Distill/
├── sharp_distill.py              # Main implementation
├── models/
│   ├── hypergraph_nn.py          # HypergraphNN with disentangled attention
│   ├── compact_gcn.py            # CompactGCN student architecture
│   ├── deberta_encoder.py        # DeBERTa with disentangled attention
│   └── distillation_mlp.py       # Knowledge distillation MLP
├── utils/
│   ├── data_loader.py            # Hypergraph construction & preprocessing
│   ├── evaluation.py             # P@K, R@K, NDCG@K, Hit Rate metrics
│   ├── contrastive_learning.py   # InfoNCE and positional encoding
│   └── knowledge_transfer.py     # Teacher-student alignment
├── experiments/
│   ├── reproduce_icml_results.py # Reproduce paper results
│   ├── ablation_studies.py       # Component contribution analysis
│   ├── hyperparameter_tuning.py  # Sensitivity analysis
│   └── scalability_analysis.py   # Performance across dataset sizes
├── data/
│   ├── amazon_cds/               # Amazon CDs dataset
│   ├── amazon_cellphones/        # Amazon Cellphones dataset  
│   ├── amazon_beauty/            # Amazon Beauty dataset
│   ├── amazon_sports/            # Amazon Sports dataset
│   └── yelp/                     # Yelp business reviews dataset
├── requirements.txt
├── LICENSE
└── README.md
```

## 🧪 Experiments

### Reproduce ICML 2025 Results

```bash
# Reproduce full paper results
python experiments/reproduce_icml_results.py

# Specific dataset evaluation
python experiments/reproduce_icml_results.py --dataset amazon_cds
python experiments/reproduce_icml_results.py --dataset yelp
```

### Ablation Studies

```bash
# Component contribution analysis  
python experiments/ablation_studies.py --component deberta
python experiments/ablation_studies.py --component contrastive_learning
python experiments/ablation_studies.py --component positional_encoding

# Knowledge transfer mechanism analysis
python experiments/ablation_studies.py --analysis knowledge_transfer
```

### Hyperparameter Sensitivity

```bash
# Temperature sensitivity analysis
python experiments/hyperparameter_tuning.py --param temperature --range 0.1,2.0

# Contrastive learning weight analysis  
python experiments/hyperparameter_tuning.py --param alpha --range 0.0,1.0

# Embedding dimension impact
python experiments/hyperparameter_tuning.py --param embedding_dim --values 64,128,256,512
```

### Custom Experiments

```python
# Example: Evaluate different DeBERTa configurations
from models.deberta_encoder import DisentangledAttention

# Test disentangled attention impact
configs = ['content_only', 'position_only', 'full_disentangled']
results = []

for config in configs:
    model = train_with_attention_config(config)
    metrics = evaluate_model(model, test_data)
    results.append(metrics)
```

## 📈 Key Innovations

### 1. Hypergraph Positional Encoding
```python
def compute_hypergraph_position(hypergraph_incidence, max_steps=3, beta=0.5):
    """
    Compute hypergraph positional features using normalized Laplacian:
    P_u = Σ(k=1 to K) β^k * (D_v^(-1/2) * H * W * D_e^(-1) * H^T * D_v^(-1/2))^k * e_u
    """
    # Compute degree matrices
    D_v = torch.sum(hypergraph_incidence, dim=1).diag()
    D_e = torch.sum(hypergraph_incidence, dim=0).diag()
    
    # Normalized Laplacian computation
    L_norm = D_v.pow(-0.5) @ hypergraph_incidence @ D_e.inverse() @ hypergraph_incidence.T @ D_v.pow(-0.5)
    
    # Multi-hop positional encoding
    pos_features = torch.zeros(num_nodes, device=hypergraph_incidence.device)
    current_hop = torch.eye(num_nodes, device=hypergraph_incidence.device)
    
    for k in range(max_steps):
        pos_features += (beta ** k) * current_hop
        current_hop = current_hop @ L_norm
        
    return pos_features
```

### 2. Temperature Annealing for Knowledge Distillation
```python
def get_temperature(self, current_epoch):
    """
    Adaptive temperature annealing:
    T(e) = T_max - (T_max - T_min) * min(1, e/E)
    """
    progress = min(1.0, current_epoch / self.total_epochs)
    return self.T_max - (self.T_max - self.T_min) * progress

def generate_soft_labels(self, embeddings, current_epoch):
    """Generate temperature-scaled soft labels for knowledge transfer"""
    temperature = self.get_temperature(current_epoch)
    scaled_logits = embeddings / temperature
    return F.softmax(scaled_logits, dim=1)
```

### 3. Multi-Modal Contrastive Learning
```python
def contrastive_learning(hgnn_embeddings, deberta_embeddings, lightgcn_embeddings):
    """
    InfoNCE-based contrastive alignment of multi-modal embeddings:
    L_con = -log(exp(sim(z_i, z_i^+)/τ) / Σ_j exp(sim(z_i, z_j)/τ))
    """
    # Normalize embeddings
    hgnn_norm = F.normalize(hgnn_embeddings, dim=1)
    deberta_norm = F.normalize(deberta_embeddings, dim=1)
    lightgcn_norm = F.normalize(lightgcn_embeddings, dim=1)
    
    # Compute contrastive losses
    user_loss = info_nce_loss(hgnn_norm, deberta_norm, temperature=0.1)
    item_loss = info_nce_loss(lightgcn_norm, deberta_norm, temperature=0.1)
    
    # Fuse embeddings
    combined_embeddings = torch.cat([hgnn_norm, deberta_norm, lightgcn_norm], dim=1)
    return combined_embeddings
```

### 4. CompactGCN Student Architecture  
```python
class CompactGCN(nn.Module):
    """
    Lightweight single-layer GCN without non-linear activations:
    Z^s = Â * X * W^s
    where Â = D^(-1/2) * (A + I) * D^(-1/2)
    """
    def __init__(self, input_dim, output_dim):
        super(CompactGCN, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        
    def forward(self, adj_matrix, features):
        # Normalized adjacency with self-loops
        adj_norm = self.normalize_adjacency(adj_matrix)
        
        # Single linear transformation without activation
        embeddings = self.linear(features)
        
        # Graph convolution
        output = adj_norm @ embeddings
        return output
```

## 📚 Citation

If you use SHARP-Distill in your research, please cite our ICML 2025 paper:

```bibtex
@inproceedings{forouzandeh2025sharp,
  title={SHARP-Distill: A 68× Faster Recommender System with Hypergraph Neural Networks and Language Models},
  author={Forouzandeh, Saman and Moradi, Parham and Jalili, Mahdi},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  pages={--},
  year={2025},
  organization={PMLR},
  venue={Vancouver, Canada}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/SHARP-Distill.git
cd SHARP-Distill

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black . && flake8 .
```

## 📋 Datasets

SHARP-Distill supports multiple real-world datasets out-of-the-box:

### Supported Datasets

| Dataset | Domain | Users | Items | Reviews | Density |
|---------|--------|-------|-------|---------|---------|
| **Amazon CDs** | Music | 71,258 | 65,443 | 1,243,755 | 17.45 |
| **Yelp** | Business | 68,754 | 48,548 | 975,910 | 14.19 |
| **Amazon Beauty** | Cosmetics | 15,152 | 10,176 | 371,345 | 24.51 |
| **Amazon Cellphones** | Electronics | 7,598 | 6,208 | 85,472 | 6.60 |
| **Amazon Sports** | Sports | 11,817 | 11,017 | 168,730 | 7.41 |

### Data Format

Your CSV file should contain the following columns:
```csv
user_id,item_id,review,rating,timestamp
user_1,item_1,"Great product! Highly recommend.",5,2023-01-01
user_1,item_2,"Not bad, could be better.",3,2023-01-02
user_2,item_1,"Excellent quality and fast delivery.",5,2023-01-03
```

### Data Preprocessing

```python
# Automatic hypergraph construction
def create_hypergraphs_and_incidence_matrices(interactions_df):
    """
    Build user and item hypergraphs from interaction data:
    - User hypergraph: Users as nodes, items as hyperedges
    - Item hypergraph: Items as nodes, users as hyperedges
    """
    users = interactions_df['user_id'].unique()
    items = interactions_df['item_id'].unique()
    
    # Create incidence matrices H_U ∈ R^{n×m}, H_I ∈ R^{m×n}
    user_hypergraph = build_user_item_incidence(users, items, interactions_df)
    item_hypergraph = build_item_user_incidence(items, users, interactions_df)
    
    return user_hypergraph, item_hypergraph
```

## 🐛 Troubleshooting

### Common Issues

**Memory Issues with Large Datasets**
```python
# Use gradient checkpointing and reduce batch size
model = HypergraphNN(..., gradient_checkpointing=True)
batch_size = 128  # Reduce from default 256

# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
```

**CUDA Out of Memory**
```python
# Monitor memory usage and adjust hypergraph depth
# Teacher HGNN layers: L=3 recommended (memory: O(R*L*d))
hypergraph_config = {
    'num_layers': 3,  # Reduce if OOM
    'hidden_dim': 64,  # Reduce if needed
    'gradient_checkpointing': True
}
```

**Slow Training on Large Datasets**
```python
# Use efficient DataLoader with multiple workers
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset, 
    batch_size=256, 
    num_workers=4, 
    pin_memory=True,
    persistent_workers=True
)
```

**DeBERTa Model Download Issues**
```python
# Pre-download DeBERTa model
from transformers import DebertaTokenizer, DebertaModel

# Manual download with cache
tokenizer = DebertaTokenizer.from_pretrained(
    'microsoft/deberta-base',
    cache_dir='./models/deberta_cache'
)
```

### Performance Optimization

**Hyperparameter Tuning Guidelines**
```python
# Optimal configurations from paper
optimal_config = {
    'temperature': 1.0,           # Teacher temperature  
    'contrastive_temp': 0.1,      # Contrastive learning temperature
    'alpha': 0.5,                 # Embedding vs positional weight
    'gamma': 0.5,                 # Student-teacher interpolation
    'learning_rate': 0.001,       # Adam optimizer
    'embedding_dim': 64           # Balance performance vs efficiency
}
```

**Training Acceleration Tips**
```python
# Enable compilation for PyTorch 2.0+
model = torch.compile(model, mode='reduce-overhead')

# Use efficient attention implementation  
torch.backends.cuda.enable_flash_sdp(True)

# Optimize for inference deployment
torch.jit.script(student_model)  # JIT compilation for production
```aler()

## 📞 Support

- 📧 **Email**: saman.forouzandeh@rmit.edu.au
- 💬 **Issues**: [GitHub Issues](https://github.com/yourusername/SHARP-Distill/issues)
- 📖 **Documentation**: [Wiki](https://github.com/yourusername/SHARP-Distill/wiki)
- 🏛️ **Institution**: RMIT University, Melbourne, Australia

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ICML 2025** reviewers for their valuable feedback and acceptance
- **RMIT University** School of Engineering for computational resources
- Built upon **PyTorch** and **HuggingFace Transformers** ecosystems
- Inspired by recent advances in **knowledge distillation** and **hypergraph neural networks**
- Special thanks to the open-source community for **DeBERTa**, **NetworkX**, and evaluation frameworks

## 🔬 Research Impact

SHARP-Distill addresses critical challenges in production recommendation systems:

- **Efficiency Gap**: Bridges the gap between research accuracy and production speed requirements
- **Multi-modal Integration**: Demonstrates effective fusion of structural and semantic information
- **Knowledge Transfer**: Advances teacher-student distillation beyond simple soft label approaches  
- **Hypergraph Applications**: Shows practical deployment of hypergraph neural networks at scale
- **Real-time Systems**: Enables complex model deployment in latency-critical applications

## 🚀 Future Work

- **Multi-language Support**: Extend DeBERTa to multilingual recommendation scenarios
- **Dynamic Hypergraphs**: Incorporate temporal evolution of user-item relationships
- **Federated Learning**: Adapt SHARP-Distill for privacy-preserving distributed training
- **Hardware Optimization**: Develop specialized kernels for CompactGCN inference
- **Domain Adaptation**: Transfer learning across different recommendation domains

---

⭐ **Star this repository if you find SHARP-Distill helpful for your research or applications!** ⭐

**Ready to deploy 68× faster recommendations? Get started with SHARP-Distill today!** 🚀
