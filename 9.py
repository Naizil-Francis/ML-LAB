import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt

heartdisease = pd.read_csv('heart.csv')
heartdisease = heartdisease.replace('?', np.nan)

print('Few examples from the dataset are given below')
print(heartdisease.head())

model = BayesianNetwork([('age', 'trestbps'), ('age', 'fbs'),
                         ('sex', 'trestbps'), ('exang', 'trestbps'),
                         ('trestbps', 'heartdisease'), ('fbs', 'heartdisease'),
                         ('heartdisease', 'restecg'), ('heartdisease', 'thalach'),
                         ('chol', 'heartdisease')])

print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartdisease, estimator=MaximumLikelihoodEstimator)

print('\nInferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)

print('\n1. Probability of HeartDisease given Age=30')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 30})
print(q.values[1])

print('\n2. Probability of HeartDisease given cholesterol=254')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': 254})
print(q.values[1])
