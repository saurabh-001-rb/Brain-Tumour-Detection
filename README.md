# Brain-Tumour-Detection
A Project Repo which Analyse Image of 3D MRI scan train and test data and predict the Tumour with the best accuracy

📌 Project Overview

This project focuses on detecting the presence of brain tumors from MRI images using Machine Learning techniques. The system analyzes medical images and classifies them into two categories: Tumor and No Tumor.

In Phase 1, we implemented a complete pipeline including data preprocessing, feature extraction, model training, and evaluation using traditional ML algorithms.

🎯 Objectives

• To study brain tumor detection using machine learning techniques

• To preprocess MRI images for better analysis

• To extract important features from brain MRI scans

• To classify MRI images as tumor or non-tumor

• To evaluate model performance using accuracy and other metrics

🧪 Dataset

The dataset consists of MRI images divided into:
Glioma
Meningioma
Pituitary
No Tumor

Tumor classes are labeled as 1, and non-tumor as 0

⚙️ Technologies Used

1. Python

2. OpenCV

3. NumPy

4. Matplotlib

5. Scikit-learn

6. Scikit-image

🔄 Methodology

1️⃣ Data Preprocessing

Images are loaded from dataset folders

Resized to 128 × 128 pixels

Converted to grayscale

2️⃣ Feature Extraction

Used HOG (Histogram of Oriented Gradients)

Extracts edge and shape-based features

Converts images into numerical feature vectors

3️⃣ Data Splitting

Dataset is split into:
80% Training
20% Testing

4️⃣ Model Training

Two machine learning models were used:

🔹 Support Vector Machine (SVM)
Kernel: Linear
Effective for high-dimensional data

🔹 Random Forest
100 decision trees
Ensemble learning method


5️⃣ Model Evaluation

Accuracy Score
Classification Report:
Precision
Recall
F1-score


📊 Results
Both models were evaluated on test data

SVM performed better, so it was selected as the final model

🧠 Prediction System

User uploads an image

Image is preprocessed (resize + grayscale)

HOG features are extracted

Model predicts:

Tumor Detected

No Tumor

🚀 Future Scope (Phase 2)

Implement Deep Learning (CNN)

Improve accuracy with larger dataset

Build a web-based interface

Real-time prediction system
