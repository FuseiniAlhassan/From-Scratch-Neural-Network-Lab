# From-Scratch-Neural-Network-Lab

**Preview Output**

![loss_anim_Two-node_hidden_layer](loss_anim_Two-node_hidden_layer.gif)

# 🔬 Formal Research Introduction:
From-Scratch Neural Network Architectures: A Comparative Study of Learning Dynamics, Optimization, and Visualization Techniques
In recent years, the interpretability and design of neural networks have become central themes in both academic research and industrial applications. While high-level frameworks like TensorFlow and PyTorch offer powerful abstractions, they often obscure the underlying mechanics of learning. This project presents a rigorous, from-scratch implementation of feedforward neural networks in Python, aimed at demystifying the core principles of neural computation, optimization, and architectural design.
The study investigates how varying network architectures specifically changes in depth and hidden layer size affect learning performance on the non-linearly separable XOR classification task. Three configurations are explored: a shallow network with a single hidden layer of two neurons, a slightly wider variant with three neurons, and a deeper model with two hidden layers. Each network is trained using mini-batch gradient descent enhanced with momentum and early stopping, allowing for robust convergence while mitigating overfitting.
To deepen understanding and improve interpretability, the project integrates dynamic visualizations of training loss and decision boundaries. These animations provide intuitive insight into how each architecture evolves during training, revealing the interplay between activation functions, weight updates, and error propagation. The use of tanh, sigmoid, and ReLU activations further enables comparative analysis of non-linear transformations and gradient behavior.
# Key contributions of this work include:
• 	A modular neural network framework built entirely from first principles, without external ML libraries.
• 	A comparative study of architectural choices and their impact on convergence speed, loss minimization, and generalization.
• 	Integration of animated visualizations to bridge the gap between abstract theory and tangible learning behavior.
• 	A reproducible pipeline for experimentation, making the project suitable for educational use, research extension, or algorithmic benchmarking.

# Overview
This project implements feedforward neural networks entirely from scratch in Python—no external machine learning libraries required. It explores how different architectures affect learning performance on the classic XOR classification task, using custom-built neurons, activation functions, and training loops.
# Features
• 	Modular neural network framework
• 	Support for , , and  activation functions
• 	Momentum-based mini-batch gradient descent
• 	Early stopping for overfitting prevention
• 	Animated visualizations of training loss and decision boundaries
• 	Architecture comparison across multiple configurations
# Architectures Explored
• 	: One hidden layer with 2 neurons
• 	: One hidden layer with 3 neurons
• 	: Two hidden layers with 2 neurons each
# Visual Outputs
All plots and animations are saved in the  directory:
• 	Static loss curves for each architecture
• 	Animated GIFs showing training loss progression
• 	Decision boundary evolution during training
Requirements
• 	Python 3.7+
• 	Numpy
• 	matpltlb
• 	Pillow  (for saving GIFs)
Install dependencies with:
pip install numpy matplotlib Pillow
# How to Run
**Simply execute the script:** main.py

# This will:
1. 	Train networks on the XOR dataset
2. 	Save static and animated plots
3. 	Compare architectures visually and numerically
# Educational Value
This lab is ideal for:
• 	Students learning neural network fundamentals
• 	Researchers exploring architecture effects
• 	Educators demonstrating learning dynamics visually
# Future Extensions
• 	Add support for multi-class classification
• 	Implement regularization (dropout, L2)
• 	Extend to real-world datasets (e.g., MNIST)
• 	Integrate adaptive optimizers (Adam, RMSprop)
