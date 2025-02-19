# 🧠 Brain Tumor Detection using SVM 🏥

## 📌 Overview
This project applies **Support Vector Machine (SVM)** for **brain tumor classification** using MRI images. The dataset contains MRI scans categorized into:
- 🟢 **Glioma Tumor**
- 🔵 **Meningioma Tumor**
- 🟡 **No Tumor**
- 🔴 **Pituitary Tumor**

## 📂 Data Source
Dataset from Kaggle:  
[🔗 Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data)

## 🔄 Project Workflow
### 1️⃣ Data Loading 📥
✅ Load MRI images from Google Drive  
✅ Resize & convert images to grayscale  
✅ Encode labels for model training  

### 2️⃣ Data Preprocessing 🛠️
✅ Normalize pixel values (0 to 1)  
✅ Split into **training & testing sets**  
✅ Standardize features using `StandardScaler`  

### 3️⃣ Model Training 🏋️
✅ Train **SVM classifier** with an **RBF kernel**  
✅ Perform **Hyperparameter Tuning** using `GridSearchCV` 🔍  

### 4️⃣ Model Evaluation 📊
✅ **Accuracy Score** ✅ **Classification Report** ✅ **Confusion Matrix**

### 5️⃣ Sample Prediction 🖼️
✅ Predict a **new MRI scan** using the trained model  

## 🎯 Results
### ✅ Model Accuracy
```
0.7538
```

### 📊 Classification Report
The classification report provides detailed metrics for model performance:
- **Precision**: Measures how many of the predicted positive cases were actually correct.
- **Recall**: Measures how well the model identified all actual positive cases.
- **F1-score**: A balance between precision and recall.
- **Support**: The number of actual instances per class.

Below is the model's classification performance:
```
                   precision    recall  f1-score   support

glioma_tumor       0.91      0.20      0.33       100
meningioma_tumor    0.69      0.98      0.81       115
no_tumor           0.69      1.00      0.81       105
pituitary_tumor    0.86      0.65      0.74        74

accuracy                                0.73       394
macro avg       0.79      0.71      0.67       394
weighted avg    0.78      0.73      0.68       394
```

### 🔢 Confusion Matrix
The confusion matrix shows how well the model classified each category:
- **True Positives (TP)**: Correctly predicted cases.
- **False Positives (FP)**: Incorrectly predicted cases that were actually from another class.
- **False Negatives (FN)**: Missed cases that the model should have predicted correctly.
- **True Negatives (TN)**: Correctly rejected cases.

The confusion matrix breakdown is shown below:
```
[[ 20  33  39   8]
 [  0 113   2   0]
 [  0   0 105   0]
 [  2  17   7  48]]
```

### 🔍 Best Hyperparameters (GridSearchCV)
```
{'C': 10, 'kernel': 'rbf'}
```

## 🚀 Installation & Usage
### ⚙️ Prerequisites
- 🐍 Python 3
- 📦 Required Libraries: `numpy`, `pandas`, `sklearn`, `opencv`, `matplotlib`, `seaborn`

### ▶️ Running the Code
1️⃣ Clone the repository:
   ```bash
   git clone https://github.com/ishmeen-11/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection
   ```
2️⃣ Upload the dataset to **Google Drive** and mount it in Colab.
3️⃣ Run the **notebook step by step**. 🚀

## 🔮 Future Improvements
✨ Use **Deep Learning models (CNNs)** for better accuracy.  
✨ Apply **Data Augmentation** to improve model generalization.  

## 👨‍💻 Author
Ishmeen Garewal ✨
