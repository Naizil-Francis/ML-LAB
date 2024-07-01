import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('opinion.csv')
print("The first 5 values of data are:\n", data.head())

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

print("\nNow the train data is:\n", X.head())

le_target = LabelEncoder()
y = le_target.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = GaussianNB()
param_grid = {'var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1.0]}
grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("\nBest parameters found:\n", grid_search.best_params_)
print("\nBest cross-validation accuracy:\n", grid_search.best_score_)
best_classifier = grid_search.best_estimator_
best_classifier.fit(X_train, y_train)
y_pred = best_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy is:", accuracy)

results = grid_search.cv_results_
plt.figure(figsize=(10, 6))
plt.plot(param_grid['var_smoothing'], results['mean_test_score'], marker='o')
plt.xscale('log')
plt.xlabel('var_smoothing')
plt.ylabel('Mean Accuracy')
plt.title('Accuracy vs. var_smoothing')
plt.grid(True)
plt.show()
