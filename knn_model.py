import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#Load Dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data,columns=data.feature_names)
y = pd.Series(data.target,name="target")

#Basic EDA

# print(X.head())
# print(X.info())
# print(y.value_counts())
# print(X.isnull().sum())


#Train / Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#Create KNN Pipeline
knn_pipeline = Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",KNeighborsClassifier())
])

#Initial KNN Model
knn_pipeline.fit(X_train,y_train)

y_pred =knn_pipeline.predict(X_test)
before_accuracy = accuracy_score(y_test,y_pred)

print("Test Accuracy: ",before_accuracy)

print("Confusion matrix: ",confusion_matrix(y_test,y_pred))

print("Classification Report: ")
print(classification_report(y_test,y_pred))