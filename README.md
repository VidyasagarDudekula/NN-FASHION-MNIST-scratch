# Fashion MNIST Neural Network (From Scratch)

This project implements a multi-layer neural network **completely from scratch in Python** using only `numpy` and `scipy`. It is designed to classify images from the **Fashion MNIST** dataset (28x28 grayscale images of clothing items) without using deep learning frameworks like PyTorch or TensorFlow.

## 🧠 Architecture

The network logic is manually implemented in `train.py`, featuring a 3-layer architecture (Input + 2 Hidden + Output) with **ReLU** activation for hidden layers and **Sigmoid** for the final output.

### Network Topology
*   **Input Layer:** 784 nodes (corresponding to 28x28 pixels)
*   **Hidden Layer 1:** 300 nodes
    *   **Activation:** ReLU (`max(0, x)`)
*   **Hidden Layer 2:** 100 nodes
    *   **Activation:** ReLU (`max(0, x)`)
*   **Output Layer:** 10 nodes (representing the 10 clothing classes)
    *   **Activation:** Sigmoid (probabilities between 0 and 1)

### Key Mathematics Implemented Manually
1.  **Forward Pass:** Matrix multiplication (`W.x`) followed by activation functions.
2.  **Backpropagation:**
    *   Calculates gradients manually using the chain rule.
    *   **ReLU Derivative:** Implemented as `1` for positive inputs and `0` for negative, preventing the vanishing gradient problem common with Sigmoid hidden layers.
3.  **Optimization:** Standard **Stochastic Gradient Descent (SGD)** with a fixed learning rate.

## 📂 Project Structure

*   `train.py`: The core neural network class (`NeuralNetwork`) and training loop.
*   `data_preprocessor.py`: Utilities for parsing CSV data and normalizing pixel values (0-255 -> 0.01-0.99).
*   `basic_stats.json`: Records accuracy metrics per epoch.

## 🚀 Performance
Early training runs show promising convergence:
*   **Epoch 0:** ~73.5% Accuracy
*   **Epoch 4:** ~81.9% Accuracy

This demonstrates that a manually built network using modern techniques like **ReLU** and a deeper architecture (2 hidden layers) can effectively learn complex patterns in image data.
