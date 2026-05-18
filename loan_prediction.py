import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

import kagglehub
from kagglehub import KaggleDatasetAdapter

warnings.filterwarnings("ignore")

file_path="train_u6lujuX_CVtuZ9i.csv"

df=kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "altruistdelhite04/loan-prediction-problem-dataset",
    file_path
)

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df['Loan_Status'].value_counts())

df['Loan_Status']=df['Loan_Status'].map({'Y':1,'N':0})

X=df.drop('Loan_Status',axis=1)
y=df['Loan_Status']

num_cols=X.select_dtypes(include=['int64','float64']).columns
cat_cols=X.select_dtypes(include=['object']).columns

numeric_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

categorical_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

preprocessor=ColumnTransformer(
    transformers=[
        ('num',numeric_transformer,num_cols),
        ('cat',categorical_transformer,cat_cols)
    ]
)

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_processed=preprocessor.fit_transform(X_train)
X_test_processed=preprocessor.transform(X_test)

selector=SelectKBest(score_func=chi2,k=5)

X_train_selected=selector.fit_transform(
    abs(X_train_processed),
    y_train
)

X_test_selected=selector.transform(
    abs(X_test_processed)
)

feature_names=preprocessor.get_feature_names_out()

selected_features=feature_names[
    selector.get_support()
]

print("\nSelected Features:")
for feature in selected_features:
    print(feature)

model=LogisticRegression(max_iter=1000)
model.fit(X_train_selected,y_train)

y_pred=model.predict(X_test_selected)
accuracy=accuracy_score(y_test,y_pred)
print("\nAccuracy Score:",accuracy)

cm=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:")
print(cm)

disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
print("\nClassification Report:\n")
print(classification_report(y_test,y_pred))

importance_scores=selector.scores_[
    selector.get_support()
]

feature_importance_df=pd.DataFrame({
    'Feature':selected_features,
    'Score':importance_scores
})

feature_importance_df=feature_importance_df.sort_values(
    by='Score',
    ascending=False
)
plt.figure(figsize=(8,5))

sns.barplot(
    x='Score',
    y='Feature',
    data=feature_importance_df
)

plt.title("Top 5 Selected Features")
plt.savefig("feature_importance.png")
plt.show()

explainer=shap.LinearExplainer(
    model,
    X_train_selected
)

shap_values=explainer.shap_values(
    X_test_selected
)

shap.summary_plot(
    shap_values,
    X_test_selected,
    feature_names=selected_features,
    show=False
)

plt.savefig("shap_summary.png")

plt.show()

with open("model_results.txt","w") as file:
    file.write("Loan Prediction Results\n\n")
    file.write(f"Accuracy Score: {accuracy}\n\n")
    file.write("Selected Features:\n")
    for feature in selected_features:
        file.write(f"- {feature}\n")
print("\nProject Completed Successfully")