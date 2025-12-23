# 🌸 DSN Replication – Deeply Supervised Neural Networks

This repository provides a **PyTorch-based replication** of the  
**Deeply-Supervised Nets (DSN) – Improving Feature Learning with Hidden Layer Supervision**.

The focus is **understanding how companion objectives enhance hidden layer discriminativeness**  
rather than purely optimizing for state-of-the-art accuracy.

- Backbone CNN with **companion heads** 🐾  
- Companion objectives supervise hidden layers early 🍄  
- Squared hinge loss ensures robust **feature learning** 🐝  
- Total loss balances output + hidden layers ✨  

**Paper reference:** [DSN – Lee et al., 2015](https://arxiv.org/abs/1409.5185) 🌷

---

## 🌌 Overview – DSN Architecture

![DSN Example](images/figmix.jpg)

### 🚀 High-level Pipeline

1. **Input image**

```math
X \in \mathbb{R}^{C \times H \times W}, \quad Z^{(0)} = X
```

2. **Backbone layers**

```math
Q^{(m)} = W^{(m)} * Z^{(m-1)}, \quad Z^{(m)} = f(Q^{(m)}), \quad m=1..M
```

3. **Companion outputs for hidden layers**

```math
\hat{y}^{(m)} = \phi(Z^{(m)}, w^{(m)}), \quad m=1..M-1
```

4. **Final output layer**

```math 
\hat{y}^{\text{out}} = \phi(Z^{(M)}, w^{\text{out}})
```

5. **Total objective**

```math
\mathcal{L}_{\text{total}} = 
\underbrace{ \|w^{\text{out}}\|^2 + L(W, w^{\text{out}}) }_{\text{output loss}} +
\sum_{m=1}^{M-1} \alpha_m \underbrace{ [ \|w^{(m)}\|^2 + \ell(W, w^{(m)}) - \gamma ]_+}_{\text{companion loss}}
```

---

## 🧠 What the Model Learns

- **Backbone**: hierarchical feature extraction 🌿  
- **Companion heads**: supervise hidden layers → discriminative features early 🍥  
- **Squared hinge loss**:

```python
loss = torch.mean(torch.clamp(1 - logits*(2*target_onehot - 1), min=0)**2)
```
- **Total los**s: weighted sum of output + companion losses 💫

- **Threshold γ**: companion loss only affects learning if above threshold
  
---
## 📦 Repository Structure

```bash
DSN-Replication/
├── src/
│   ├── layers/
│   │   ├── conv_block.py            # Reusable Conv + activation block for feature extraction
│   │   ├── activation.py            # Activation functions (ReLU, LeakyReLU, Sigmoid, etc.)
│   │   ├── normalization.py         # Normalization layers (BatchNorm, LayerNorm)
│   │   └── pooling.py               # Pooling operations (MaxPool, AvgPool)
│   │
│   ├── companions/
│   │   ├── companion_head.py        # Companion classifier (SVM/Softmax) for hidden layers
│   │   └── companion_loss.py        # Squared hinge loss for companion objectives
│   │
│   ├── backbone/
│   │   ├── backbone_block.py        # Main CNN blocks (Conv + Pooling layers)
│   │   └── feature_map.py           # Utilities to manage intermediate feature maps
│   │
│   ├── model/
│   │   └── dsn_net.py               # Full model: backbone + companion heads + output head
│   │
│   ├── loss/
│   │   └── total_loss.py            # Combine output loss + companion losses (weighted, thresholded)
│   │
│   └── config.py                    # Hyperparameters: α_m, γ, number of layers, output/head types
│
├── images/
│   └── figmix.jpg         
│
├── requirements.txt                
└── README.md        

```
---


## 🔗 Feedback

For questions or feedback, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
