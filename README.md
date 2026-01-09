# Fashion MNIST Classification 👗👟

> **From Scratch Neural Network to CNN: A Deep Learning Evolution**

This project demonstrates the power of convolutional neural networks by comparing a **custom-built fully connected neural network** against a **modern CNN architecture** on the Fashion MNIST dataset. Both implementations achieve strong results, with the CNN delivering a significant accuracy boost.

---

## 🎯 Results at a Glance

| Model | Test Accuracy | Test Loss | Architecture |
|:------|:-------------:|:---------:|:-------------|
| **Fully Connected NN** (from scratch) | 87.48% | 0.2177 | 784 → 300 → 100 → 10 |
| **CNN** (PyTorch) | **93.19%** | 0.2689 | Conv2D + Deep FC + Dropout + LayerNorm |

### 📈 Model Evolution & Performance Journey

| Version | Model | Accuracy | Loss | What Changed | Why It Helped |
|:-------:|:------|:--------:|:----:|:-------------|:--------------|
| **v0** | Fully Connected NN | 87.48% | 0.2177 | *Baseline* — from scratch with NumPy | — |
| **v1** | CNN (Basic) | 91.73% | 0.2783 | Added Conv2D layers + BatchNorm + MaxPool | Spatial feature extraction captures patterns FC layers miss |
| **v2** | CNN + Deep FC Head | 92.54% | 0.2860 | Added 2× hidden FC layers (1024) + LayerNorm | Deeper classification head learns more complex decision boundaries |
| **v3** | CNN + Full Regularization | **93.19%** | **0.2689** | Added Dropout(0.1) to FC layers | Prevents overfitting, improves generalization on test data |

```diff
  v0 (FC NN)     → 87.48%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▌
+ v1 (CNN)       → 91.73%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▌ (+4.25%)
+ v2 (Deep FC)   → 92.54%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▌ (+0.81%)
+ v3 (Dropout)   → 93.19%  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▌ (+0.65%)
```

**Total improvement: +5.71% accuracy** through iterative architectural enhancements.

---

## 📉 Training Loss Curve (CNN)

![Training vs Validation Loss](training_loss_plot.png)

The loss curve shows:
- **Rapid convergence** in the first ~500 steps
- **Excellent generalization** — validation loss stays below training loss throughout
- **No overfitting** — thanks to aggressive dropout regularization across all layers
- **Final validation loss** settles around ~0.20

---

## 🧠 Model Architectures

### 1. Fully Connected Neural Network (`train.py`)
*Built completely from scratch using only NumPy and SciPy*

```
Input (784) → Hidden1 (300, ReLU) → Hidden2 (100, ReLU) → Output (10, Sigmoid)
```

**Key Features:**
- ✅ Manual forward/backward propagation
- ✅ Hand-coded ReLU derivatives
- ✅ Stochastic Gradient Descent (SGD)
- ✅ No deep learning frameworks

**Highlights:**
- Implements **backpropagation from scratch** using the chain rule
- Uses **ReLU activations** in hidden layers to prevent vanishing gradients
- Achieves **87.48% accuracy** with pure NumPy math

---

### 2. Convolutional Neural Network (`cnn_train.py`)
*Built with PyTorch for GPU-accelerated training*

```
Input (1×28×28)
    ↓
┌─────────────────────────────────────────────────────────────┐
│  CONVOLUTIONAL FEATURE EXTRACTOR                            │
├─────────────────────────────────────────────────────────────┤
│  Conv2D (64 filters, 3×3, padding=1)                        │
│      → ReLU → BatchNorm2D → MaxPool (2×2) → Dropout(0.1)    │
│                                                             │
│  Conv2D (128 filters, 3×3)                                  │
│      → ReLU → BatchNorm2D → MaxPool (2×2) → Dropout(0.1)    │
└─────────────────────────────────────────────────────────────┘
    ↓
  Flatten (128×6×6 = 4608)
    ↓
┌─────────────────────────────────────────────────────────────┐
│  FULLY CONNECTED CLASSIFICATION HEAD                        │
├─────────────────────────────────────────────────────────────┤
│  Linear (4608 → 1024) → ReLU → LayerNorm → Dropout(0.1)     │
│  Linear (1024 → 1024) → ReLU → LayerNorm → Dropout(0.1)     │
│  Linear (1024 → 10)   [Output logits]                       │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- ✅ 2 Convolutional layers with increasing filter depth (64 → 128)
- ✅ Batch Normalization after conv layers for training stability
- ✅ **2 Hidden fully connected layers** (1024 neurons each)
- ✅ **Layer Normalization** after FC layers for improved gradient flow
- ✅ **Dropout (0.1) on ALL layers** — conv blocks AND FC layers
- ✅ AdamW optimizer with learning rate 3e-4
- ✅ Cross-Entropy loss function
- ✅ MPS/CUDA acceleration support

**Highlights:**
- Achieves **93.19% accuracy** on the test set
- Trains for 10 epochs with 90/10 train-validation split
- Comprehensive dropout regularization prevents overfitting
- Layer normalization enables stable training of deeper networks

---

## 📂 Project Structure

```
fashion-mnist-scratch/
├── train.py                 # From-scratch fully connected NN
├── cnn_train.py             # PyTorch CNN implementation
├── data_preprocessor.py     # Data loading & normalization utilities
├── normal_nn_stats.json     # FC network evaluation results
├── cnn_model_stats.json     # CNN evaluation results
├── training_loss_plot.png   # Loss curve visualization
└── README.md
```

---

## 🚀 Quick Start

### Train the Fully Connected Network
```bash
python train.py
```

### Train the CNN
```bash
python cnn_train.py
```

*Requires Fashion MNIST CSV files (`fashion-mnist_train.csv`, `fashion-mnist_test.csv`) in the project directory.*

---

## 🔬 Why the CNN Wins

| Aspect | Fully Connected NN | CNN |
|--------|-------------------|-----|
| **Spatial Awareness** | Treats pixels independently | Learns local patterns (edges, textures) |
| **Parameter Efficiency** | ~266K params (dense) | Shared conv filters + deep FC head |
| **Translation Invariance** | ❌ No | ✅ Yes |
| **Feature Hierarchy** | Flat representation | Low→High level features |
| **Normalization** | None | BatchNorm + LayerNorm |
| **Regularization** | None | **Dropout on ALL layers** |

The CNN architecture combines:
1. **Convolutional feature extraction** — captures spatial patterns efficiently
2. **Deep fully connected head** — provides powerful non-linear classification
3. **Aggressive regularization** — dropout on both conv and FC layers prevents overfitting
4. **Modern normalization** — BatchNorm + LayerNorm for stable, fast training

---

## 📊 Dataset

**Fashion MNIST** consists of 70,000 grayscale images (28×28 pixels) across 10 clothing categories:

| Label | Class |
|:-----:|:------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## 🛠️ Dependencies

- **NumPy** & **SciPy** (for from-scratch NN)
- **PyTorch** (for CNN)
- **Matplotlib** (for loss visualization)

---

<p align="center">
  <i>Built to understand neural networks from the ground up 🧪</i>
</p>
