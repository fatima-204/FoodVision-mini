Food Vision Mini 🍕🥩🍣
Food Vision Mini is a food classification model that categorizes images into three classes: Pizza, Steak, and Sushi. It uses an EfficientNet-B2 model as a feature extractor to achieve quick and accurate classification. This project showcases the power of deep learning and computer vision.

Features
Image Classification: Classifies images of food into three categories: Pizza, Steak, and Sushi.

EfficientNet-B2: Utilizes a pre-trained EfficientNet-B2 model for feature extraction and fine-tuned to classify the three types of food.

Real-time Predictions: Users can upload an image of food and get instant predictions with a display of classification probabilities and prediction time.

Technologies Used
Python

PyTorch (for model building and inference)

Gradio (for creating the user interface)

PIL (Python Imaging Library for image manipulation)

EfficientNet-B2 (pre-trained deep learning model for feature extraction)

Dataset
This project uses a small set of food images for demonstration purposes (Pizza, Steak, and Sushi). You can add more images to the examples/ folder to test the classifier further.

Installation
Prerequisites
Python 3.x

PyTorch

Gradio

PIL

torchvision

Model Architecture
The model is based on EfficientNet-B2, a lightweight and highly efficient deep learning model. The classifier head is modified to output probabilities for three classes: Pizza, Steak, and Sushi.

Key Layers:
EfficientNet-B2 feature extractor.

Dropout layer for regularization.

Fully connected layer that maps the features to the class probabilities.
Acknowledgments
Kaggle for providing datasets (if you use any dataset from Kaggle).

Gradio for making the interface creation easy.

PyTorch and EfficientNet for the model architecture and pre-trained weights.
