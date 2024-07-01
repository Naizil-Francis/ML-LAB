import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the digits dataset and segregating targets and values
digits = load_digits()
X = digits.data
y = digits.target

# Create a DataFrame to visualize the data
digits_df = pd.DataFrame(data=X)
digits_df['target'] = y
print(digits_df.head())

# Split the dataset into test and train and standardize the datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM Model and predict for the required data
svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)

# Evaluate the ML model
accuracy = accuracy_score(y_test, y_pred)
cl = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Printing results
print("Accuracy:\n", accuracy * 100)
print("\nClassification Report:\n", cl)
print("\nConfusion Matrix:\n", conf_matrix)
