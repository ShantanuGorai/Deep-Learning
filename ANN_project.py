import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\ACER\Downloads\archive (2)\Iris.csv")
print(df['Species'].value_counts())
print(df.info())

X = df.drop(columns = ['Species', 'Id'])
y = df['Species']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, shuffle=True, stratify= y_encoded)

scaler = StandardScaler()
X_train_Scaled = scaler.fit_transform(X_train)
X_test_Scaled = scaler.fit_transform(X_test)

per = Perceptron(max_iter=1000, random_state=42)
per.fit(X_train_Scaled, y_train)

y_pred_percep = per.predict(X_test_Scaled)
accuracy = accuracy_score(y_test, y_pred_percep)
print("Accuracy Score : ", accuracy)




