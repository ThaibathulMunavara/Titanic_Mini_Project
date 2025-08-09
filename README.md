# Titanic Survival Prediction 🚢

A machine learning project that predicts survival outcomes of passengers on the Titanic using various classification techniques. This project demonstrates the full lifecycle of a data science pipeline — from data cleaning and exploratory analysis to model training, hyperparameter tuning and and evaluation.
The analysis focuses on:
- Exploring **patterns** in passenger demographics and travel details
- Detecting **anomalies and outliers**
- Identifying **key features** that influence survival
- Building and evaluating **machine learning models** to predict survival

-----------------------------------------------------------------------------------------------------------------------------------

## 📊 Dataset Overview

- **Source:** [Kaggle Titanic Dataset]([https://www.kaggle.com/datasets/yasserh/titanic-dataset/data])
- **Rows:** 891 passengers  
- **Columns:** 12 features including demographics, ticket details, and survival outcome
  
    | Column Name | Description | Type |
    --------------------------------------------------------
    | PassengerId | Unique passenger identifier | Integer |
  
    | Survived    | Survival (0 = No, 1 = Yes) | Integer |
  
    | Pclass      | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) | Integer |
  
    | Name        | Passenger name | String |
  
    | Sex         | Gender | String |
  
    | Age         | Age in years | Float |
  
    | SibSp       | # of siblings/spouses aboard | Integer |
  
    | Parch       | # of parents/children aboard | Integer |
  
    | Ticket      | Ticket number | String |
  
    | Fare        | Passenger fare | Float |
  
    | Cabin       | Cabin number | String |
  
    | Embarked    | Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton) | String |

- **Target Variable:** `Survived` (0 = No, 1 = Yes)

-----------------------------------------------------------------------------------------------------------------------------------

## 📌 Project Objectives

- Perform exploratory data analysis (EDA) to understand the data and relationships
- Handle missing values and outliers
- Encode categorical variables
- Balance the dataset using **SMOTE**
- Scale features for model optimization
- Build and evaluate a classification model

------------------------------------------------------------------------------------------------------------------------------------

## ⚙️ Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Imbalanced-learn (for SMOTE)

-----------------------------------------------------------------------------------------------------------------------------------

## 🧹 Data Preprocessing

- **Missing Values:** Imputed `Age` with median, `Embarked` with mode, dropped `Cabin`,`PassengerId`, `Name`, `Ticket` as a part of removing irrelevant or unnecessary data to improve model efficiency and reduce noise. 
- **Categorical Encoding:** Label encoding for `Sex`and `Embarked`   
- **Outlier Detection:** Used boxplots to identify outliers in `Fare` and `Age`, treated extreme values  
- **Feature Scaling:** StandardScaler used on `Age` and `Fare`  
- **Class Balancing:** Applied SMOTE to handle imbalance in `Survived` classes

-----------------------------------------------------------------------------------------------------------------------------------

## 📈 Exploratory Data Analysis (EDA)
-  Explored:
      - **Summary statistics** (mean, median, std, etc.)
      - **Distribution plots** (histograms, boxplots)
      - **Correlation analysis** (heatmaps, pairplots)
      - **Outlier detection** (IQR method)
      - **Categorical analysis** (bar charts for `Sex`, `Pclass`, etc.)
- Identified key features affecting survival:
  - **Sex**: Males and females have equal probabilities of survival,ensuring that the model was not biased toward gender.
  - **Pclass**: 1st class had higher survival rate
  - **Age**: Younger passengers had higher survival probability
- Visualizations included histograms, heatmaps, boxplots, and survival ratios

-----------------------------------------------------------------------------------------------------------------------------------

## 🤖 Model Training and Evaluation

- **Algorithm Used:** Random Forest Classifier
- **Data Split:** Train-test split using scikit-learn (typically 80/20 or 70/30)
- **Evaluation Metrics:**
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
  - **Confusion Matrix**: Used to visualize the comparison between actual and predicted labels, helping to evaluate true positives, false positives, true negatives, and false negatives.

- **Result:** Achieved strong predictive performance with well-balanced precision and recall after applying SMOTE for class balance.

-----------------------------------------------------------------------------------------------------------------------------------

## 💡 Real-World Relevance

The techniques used in this project simulate real-world classification tasks such as:

- **Customer churn prediction**  
- **Loan default risk modeling**  
- **Medical diagnosis classification**

------------------------------------------------------------------------------------------------------------------------------------

## 📁 Project Structure

├── data/ # Dataset (Titanic-Dataset.csv)

├── EDA.ipynb/ # Jupyter notebooks for EDA & modeling

├── titanic.ipynb # Main script for model training

├── README.md # Titanic Project documentation

-----------------------------------------------------------------------------------------------------------------------------------

## ⚙️ Installation

### Make sure you have **Python 3** and **Jupyter Notebook** installed on your system.
```bash
### Clone the repository
  -git clone https://github.com/ThaibathulMunavara/Titanic.ipynb
  -cd Titanic

### Install dependencies

  pip install -r requirements.txt

--------------------------------------------------------------------------------------------------------------------------------------

## 🚀 **How to run**

- After Installation, **Open in Terminal / Command Prompt**
   --Navigate to the project folder:
       ```bash
       cd path/to/project
       ```
- **Run the EDA Notebook**
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open `Titanic.ipynb` or `EDA.ipynb`  and run all cells.
----------------------------------------------------------------------------------------------------------------------------------------

