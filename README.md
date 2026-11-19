# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries and load the dataset containing email messages and their labels (Spam or Ham).

2.Preprocess the data by converting text messages into numerical form using TF-IDF vectorization and splitting the dataset into training and testing sets.

3.Train the Support Vector Machine (SVM) classifier using the training dataset to learn the patterns that distinguish spam emails from normal emails.

4.Test the trained model and evaluate performance by predicting the labels of the test dataset and calculating the accuracy of the classifier.

## Program:
```python
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: JOANNE JAISON
RegisterNumber: 212225020017
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("C:/Users/acer/Downloads/spam.csv", encoding='latin-1')

# Select required columns
data = data[['v1','v2']]
data.columns = ['label','message']

# Convert labels into numbers
data['label'] = data['label'].map({'ham':0,'spam':1})

# Features and target
X = data['message']
y = data['label']

# Convert text to numeric using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train SVM model
model = SVC(kernel='linear')
model.fit(X_train,y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test,y_pred)

print("Accuracy:",accuracy)

print("\nClassification Report:")
print(classification_report(y_test,y_pred))
```

## Output:
<img width="1195" height="272" alt="Screenshot 2026-11-19 164219" src="https://github.com/user-attachments/assets/ea3feea9-6d4a-4bc3-8167-2c2d608e47c7" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
