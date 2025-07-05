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
