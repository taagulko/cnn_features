Deterministic CNN Feature Extractor & Visualizer
A lightweight desktop application built with Python to extract, normalize, and visualize interpretable geometric feature vectors from images using a deterministic 3-layer Convolutional Neural Network (CNN) without deep learning framework overhead.

Overview
Traditional deep CNNs often act as black boxes and require extensive training pipelines. This project explores a deterministic, training-free approach to image feature representation.

The application implements a 3-layer CNN architecture directly using NumPy:

Layer 1 (Preprocessing & Patching): Resizes the input image (256x256), converts it to grayscale, performs threshold binarization, and divides it into local patches.

Layer 2 (Feature Extraction & Convolution): Convolves each patch against a bank of 8 predefined directional geometric kernels (vertical, horizontal, and diagonal orientations) to compute activation scores.

Layer 3 (Classification & Aggregation): Aggregates local scores into a normalized feature vector and classifies the dominant structure into 4 categories (Vertical, Horizontal, Diagonal, or Mixed).

Features
Built from Scratch: Core mathematical operations (2D convolution, matching score calculation, vector normalization) implemented via NumPy without PyTorch/TensorFlow.

Configurable Patch Resolution: Switch dynamically between patch sizes (4x4, 8x8, 16x16, 32x32) to evaluate spatial granularity vs. noise invariance.

Rich Real-Time GUI: Modern dark-themed interface built using Tkinter and embedded Matplotlib canvases displaying 6 analytical visualizer plots simultaneously.

Synthetic Noise Generator: Built-in tool to generate test patterns and assess model behavior on noisy geometric shapes.

Tech Stack
Language: Python 3.9+

Numerical Computing: NumPy

Image Processing: Pillow (PIL)

Visualization: Matplotlib

GUI: Tkinter

Getting Started
Clone the repository:
git clone https://github.com/taagulko/cnn_features.git

Install dependencies:
pip install numpy pillow matplotlib

Run the application:
python app.py

Author
Developed as an academic and research project at Chernivtsi National University (Computer Science).
