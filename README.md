# K-Neighbours
# Plant Disease Identification using Deep Learning

## Overview
This project focuses on building a **custom Convolutional Neural Network (CNN)** for **plant disease identification**. The model is trained using **image classification techniques** to detect and classify various plant diseases based on leaf images. The project also leverages **image processing techniques** like OpenCV and Pillow for preprocessing.

## Features
- **Custom CNN Architecture**: Optimized for image classification.
- **Image Processing**: Uses OpenCV and Pillow for data augmentation and preprocessing.
- **Small to Mid-Sized Dataset**: Trained on a carefully curated dataset of diseased and healthy plant images.
- **PyTorch Implementation**: Utilizes PyTorch for deep learning model training and inference.

## Requirements
- Python 3.x
- PyTorch
- OpenCV
- Pillow
- NumPy
- Matplotlib

## Installation
```md
pip install torch torchvision opencv-python pillow numpy matplotlib
```

## Usage
1. **Preprocess Data**: Resize and normalize images.
2. **Train the Model**: Run the training script to train the CNN model.
3. **Evaluate Performance**: Test the model on validation and test datasets.
4. **Inference**: Use the trained model to predict plant diseases from new images.

## Dataset
The dataset consists of images of various plant leaves labeled with disease categories. Ensure proper dataset structuring for effective training.

## Future Improvements
- **Enhancing Dataset**: Adding more images for better accuracy.
- **Hyperparameter Tuning**: Optimizing learning rate, batch size, etc.
- **Deployment**: Creating a web or mobile application for real-world usability.

## License
This project is open-source and available for educational and research purposes.

