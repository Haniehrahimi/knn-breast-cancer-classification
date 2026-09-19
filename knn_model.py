import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt

#Load Dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data,columns=data.feature_names)
y = pd.Series(data.target,name="target")

#Basic EDA

print(X.head())
X.info()
print(y.value_counts())
print(X.isnull().sum())


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

#Cross Validation
cv_scores = cross_val_score(knn_pipeline,X_train,y_train,cv=5,scoring="accuracy")

print("CV Scores: ",cv_scores)
print("Mean CV Accuracy: ",cv_scores.mean())
print("Std CV Accuracy: ",cv_scores.std())

#GridSearchCV
param_grid = {
    "classifier__n_neighbors":[3,5,7,9,11,13,15],
    "classifier__weights":["uniform","distance"]
}

grid = GridSearchCV(estimator=knn_pipeline,param_grid=param_grid,cv=5,scoring="accuracy",n_jobs=-1)
grid.fit(X_train,y_train)

#Best Parameters
print("GridSearchCv Results")
print("Best Params: ",grid.best_params_)
print("Best CV Accuracy: ",grid.best_score_)

#Final Model
best_model = grid.best_estimator_ 
y_pred_final = best_model.predict(X_test) 
after_accuracy = accuracy_score(y_test, y_pred_final)

#Final Model Evaluation
print("Final Test Accuracy:", after_accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

print("\nClassification Report:") 
print(classification_report(y_test,y_pred_final))

# Before vs After Comparison
print(f"Before GridSearchCV: " f"{before_accuracy:.4f}")
print(f"After GridSearchCV: " f"{after_accuracy:.4f}")
improvement = (after_accuracy - before_accuracy) 
print(f"Improvement: " f"{improvement:.4f}")

# Before vs After Plot

labels = ["Before GridSearchCV","After GridSearchCV"]
accuracies = [before_accuracy * 100, after_accuracy * 100 ] 
plt.figure(figsize=(8, 5)) 
bars = plt.bar(labels,accuracies)

plt.title( "KNN Accuracy: Before vs After GridSearchCV")
plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)") 
plt.ylim(90, 100) 

for bar, accuracy in zip( bars, accuracies ):
    plt.text( bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{accuracy:.2f}%", ha="center" ) 


plt.tight_layout() 
plt.show()