import pandas as pd
from sklearn.datasets import load_breast_cancer

#Load Dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data,columns=data.feature_names)
y = pd.Series(data.target,name="target")

#Basic EDA

# print(X.head())
# print(X.info())
# print(y.value_counts())
# print(X.isnull().sum())


