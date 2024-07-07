from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np

#load dataset and get features and data
wine = load_wine()
X = wine.data
y = wine.target

#split the dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#define Decision Tree classifier
clf = DecisionTreeClassifier(
    criterion='entropy',
    splitter='best',
    max_depth=3,
    min_samples_split=6,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42
)
clf.fit(X_train, y_train)#fit training data

y_pred = clf.predict(X_test)#predict the data using test data
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")#correct values
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=wine.feature_names, class_names=wine.target_names, filled=True)
plt.show()
