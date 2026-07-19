import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("dataset/human_vital_signs.csv")

# Keep required columns
df = df[["Heart Rate", "Risk Category"]]

# Encode labels
encoder = LabelEncoder()
df["Risk Category"] = encoder.fit_transform(df["Risk Category"])

print(df.head())

print("\nClasses:")
print(encoder.classes_)


X = df[["Heart Rate"]]

y = df["Risk Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

predictions = model.predict(X_test)

print(predictions[:20])


accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)


joblib.dump(model, "models/risk_model.pkl")

print("\nModel Saved Successfully!")