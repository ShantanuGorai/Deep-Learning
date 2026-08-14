import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")

df = pd.read_csv(
    r"C:\Users\ACER\Downloads\archive (2)\Iris.csv"
)

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

print(df["Species"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x="Species", data=df)
plt.title("Iris Species Distribution")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="PetalLengthCm",
    y="PetalWidthCm",
    hue="Species",
    s=80
)
plt.title("Petal Length vs Petal Width")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

X = df.drop(columns=["Species", "Id"])
y = df["Species"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(16, activation="relu", input_shape=(4,)),
    Dropout(0.2),
    Dense(12, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test Accuracy:", accuracy)
print("Test Loss:", loss)

y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)
y_actual = np.argmax(y_test, axis=1)

print("\nClassification Report")
print(
    classification_report(
        y_actual,
        y_pred,
        target_names=encoder.classes_
    )
)

cm = confusion_matrix(y_actual, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

sample = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.7, 3.0, 5.2, 2.3]
])

sample_scaled = scaler.transform(sample)

predictions = model.predict(sample_scaled)
predicted_classes = encoder.inverse_transform(
    np.argmax(predictions, axis=1)
)

print("\nSample Predictions")

for i, prediction in enumerate(predicted_classes):
    print(sample[i], "->", prediction)