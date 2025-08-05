#  Diabetes Prediction using Machine Learning

This project demonstrates how to build a Logistic Regression model to predict whether a patient has diabetes based on medical attributes. It uses the Pima Indian Diabetes dataset and is designed for Google Colab, including integration with Google Drive.



# Dataset

The dataset used is the **Pima Indians Diabetes Database**. It contains 768 entries with 8 clinical features and 1 binary outcome:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome` (0 = No diabetes, 1 = Diabetes)

Location in Google Drive:  
`MyDrive/CSV FILE/diabetes.csv`



# Objectives

- Load dataset from Google Drive
- Handle missing values and perform EDA
- Train a Logistic Regression model
- Evaluate performance using classification metrics



# Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Google Colab
- Google Drive


# How to Run

# In Google Colab
1. Mount your Google Drive:
   
from google.colab import drive
drive.mount('/content/drive')

2. Load the dataset:
csv_path = '/content/drive/MyDrive/CSV FILE/diabetes.csv'
df = pd.read_csv(csv_path)

3. Run the notebook cells for EDA, cleaning, training, and evaluation.
# KNN from Scratch: Flower and News Classification with Custom Evaluation Metrics

## 📌 Project Title
**KNN from Scratch: Flower and News Classification with Custom Evaluation Metrics**

## 🧠 Overview
This project implements the **K-Nearest Neighbors (KNN)** algorithm from scratch using Python. It also includes the development of custom evaluation metrics such as accuracy, confusion matrix, precision, recall, and F1-score.

The algorithm is tested on the **Wisconsin Breast Cancer Dataset** (CSV file: `KNNAlgorithmDataset.csv`). The aim is to classify tumors as benign or malignant, optimize model performance, and compare the custom implementation with scikit-learn's `KNeighborsClassifier`.

## 🎯 Objectives
- ✅ Implement KNN algorithm from scratch (without ML libraries)
- ✅ Build custom evaluation metrics:
  - Accuracy
  - Confusion Matrix
  - Precision
  - Recall
  - F1-Score
- ✅ Apply the algorithm to a real dataset
- ✅ Optimize hyperparameter `k` and train-test split ratio
- ✅ Compare with `sklearn` implementation
- ✅ Document mathematical theory behind KNN and metrics

## 🧪 Procedure
1. **Theoretical Foundation**:
   - Review Euclidean distance formula
   - Define evaluation metric formulas using TP, TN, FP, FN

2. **KNN Class Implementation**:
   - `fit(X, y)` – store training data
   - `predict(X)` – classify test points using majority vote of `k` nearest neighbors

3. **Metric Functions**:
   - Custom functions for all evaluation metrics

4. **Data Preprocessing**:
   - Drop unused columns (`id`, `Unnamed: 32`)
   - Encode target (`M`→1, `B`→0)
   - Scale features using `StandardScaler`

5. **Model Optimization**:
   - Find best value of `k` (1 to 25)
   - Try various train-test split ratios (e.g., 80/20, 75/25)

6. **Final Comparison**:
   - Train both custom and sklearn KNN on same data
   - Evaluate using all metrics

## 📂 Files
- `KNNAlgorithmDataset.csv` – Breast Cancer Dataset
- `knn_from_scratch.ipynb` – Jupyter Notebook implementation
- `LabReport02-knnFromScratch.pdf` – Project report

## 📈 Output Summary
- **Accuracy**: 95.6%
- **Precision**: 92.9%
- **Recall**: 95.3%
- **F1 Score**: 94.1%

The custom KNN model performs nearly identically to scikit-learn’s, proving the correctness and effectiveness of the implementation.

## 🧑‍💻 Author
- **Name**: Al Motakabbir Mahmud Shihab  
- **ID**: 222002061  
- **University**: Green University of Bangladesh  
- **Course**: CSE 412 (Machine Learning Lab), Section: 222 D3  

## 🔗 GitHub Repository
[🔗 GitHub Notebook Link](https://github.com/SHIBU-69/Labreport_/blob/main/222002061_CSE312_222D3_LabReport02_knnFromScratch.ipynb)
