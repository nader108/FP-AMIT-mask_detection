Mask Detection Model - Final Project for AMIT
Project Overview
This project implements a Face Mask Detection Model using Convolutional Neural Networks (CNN) with the MobileNetV2 architecture. The model is trained to classify images into two categories: "With Mask" and "Without Mask." It is designed to operate in real-time for detecting mask usage, especially relevant in the context of health protocols.

Key Features
Real-Time Mask Detection: The model can identify individuals wearing masks or not, which is critical in enforcing safety protocols.
Deep Learning Model: Built using MobileNetV2, a lightweight CNN architecture pre-trained on ImageNet, and fine-tuned for the mask detection task.
Data Augmentation: Implemented techniques like rotation, zoom, and horizontal flipping to improve model robustness.
High Accuracy: Achieved 99% accuracy in classifying mask-wearing individuals based on a test dataset.
Technologies Used
TensorFlow & Keras: For building, training, and evaluating the deep learning model.
MobileNetV2: A pre-trained model used as the base for feature extraction, with a custom head for mask detection.
OpenCV: For image loading, processing, and augmentations.
Matplotlib: For visualizing the training progress and evaluation metrics.
Scikit-learn: For generating classification reports and evaluating model performance.
Model Training
The model was trained for 20 epochs with a batch size of 32. The Adam optimizer was used with a learning rate of 1e-4, and the model achieved an accuracy of 99% on the validation set after training.

How to Use
Clone the repository:
bash
Copy code
git clone https://github.com/nader108/final-project-for-AMIT-mask_detection_model
Install the required libraries:
bash
Copy code
pip install -r requirements.txt
Train the model using the mask_detection_model.ipynb notebook or use the trained model for prediction:
bash
Copy code
python mask_detection_model.py
Model Evaluation
The model's performance was evaluated using the classification report, achieving:

Precision: 99% for both "With Mask" and "Without Mask"
Recall: 99% for both classes
F1-Score: 99% for both classes
Accuracy: 99%
Future Work
Extend the model to detect other personal protective equipment (PPE) such as face shields.
Implement a real-time video feed detection system for public spaces.
Enhance the dataset by adding images from diverse environments to improve model generalization.
Model Output
The final trained model is saved as mask_detection_model_2.h5.

