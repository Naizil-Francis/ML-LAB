

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


# Load the digits dataset and segregating targets and values

# In[50]:


digits = load_digits()
X = digits.data
y = digits.target


# Set the target values and print headers

# In[51]:


digits_df = pd.DataFrame(data=X)
digits_df['target'] = y
print(digits_df.head())


# Split the dataset into test and train and standardize the datasets

# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train SVM Model and predict for the required data

# In[53]:


svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)


# Evaluate the ML model

# In[54]:


accuracy = accuracy_score(y_test, y_pred)
cl=classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


# Printing part

# In[55]:


print("Accuracy:\n", accuracy*100)
print("\nClassification Report:\n",cl)
print("\nConfusion Matrix:\n",conf_matrix)


# In[ ]:




