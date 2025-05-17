import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
data_dir = os.path.join(project_dir, 'data')
file_path = os.path.join(data_dir, 'my_vc_data.csv')
df = pd.read_csv(file_path)

features = [
    'got_funding_during_recession',
    'amount_milestones_corrected',
    'is_founding_year_recession_year',
    'NumContributors',
    'NumPartners',
    'biggest_funding_amount',
    'age_2014_in_years',
    'BusinessField',
    'Funding_total'
]
target = 'Is_Successful'

df = df.dropna(subset=[target])
X = df[features]
y = df[target].astype(int)

categorical = ['BusinessField']
numeric = [f for f in features if f not in categorical]

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical)
    ],
    remainder='passthrough'
)

rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=10,
    min_samples_split=5,
    random_state=25
)

pipeline = Pipeline(steps=[  
    ('preprocessor', preprocessor),
    ('classifier', rf)
])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=50)

y_proba = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')[:, 1]
y_pred = (y_proba >= 0.58).astype(int)

auc = roc_auc_score(y, y_proba)  
f1 = f1_score(y, y_pred) 
acc = accuracy_score(y, y_pred)  
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)

print(f"AUC:  {auc:.4f}")
print(f"F1:   {f1:.4f}")
print(f"ACC:  {acc:.4f}")
print(f"Prec: {prec:.4f}")
print(f"Rec:  {rec:.4f}")
print("\nClassification Report:")
print(classification_report(y, y_pred))

cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()
