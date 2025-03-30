import os
import warnings
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, roc_curve,
                             precision_recall_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')

# ------------------------------
# Pfaddefinitionen
# ------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
data_dir = os.path.join(project_dir, 'data')
results_dir = os.path.join(project_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

file_path = os.path.join(data_dir, 'Exported_prepared_data_cleaned.csv')
print(f"Lade Daten aus {file_path}...")

# ------------------------------
# Daten einlesen
# ------------------------------
try:
    df = pd.read_csv(file_path, sep=',', engine='python', on_bad_lines='warn')
    print(f"Datensatz-Form: {df.shape}")

    # Falls nur eine Spalte erkannt wird, alternative Trennzeichen versuchen
    if df.shape[1] == 1:
        print("Nur eine Spalte erkannt, versuche andere Trennzeichen...")
        for sep in [',', ';', '\t', '|']:
            try:
                temp_df = pd.read_csv(file_path, sep=sep, engine='python', on_bad_lines='warn')
                print(f"Datensatz-Form mit '{sep}': {temp_df.shape}")
                if temp_df.shape[1] > 1:
                    df = temp_df
                    print("Mehrere Spalten erkannt!")
                    break
            except Exception as e:
                print(f"Fehler mit Trennzeichen '{sep}': {str(e)}")

except Exception as e:
    print(f"Fehler beim Einlesen der Daten: {str(e)}")
    raise

print("\nVerfügbare Spalten:")
print(df.columns.tolist())

# ------------------------------
# Zielvariable: is_successful (1 = erfolgreich, 0 = Failed)
# ------------------------------
if 'Status' in df.columns and 'is_successful' not in df.columns:
    df['is_successful'] = df['Status'].apply(lambda x: 1 if 'Successful' in str(x) else 0)
elif 'is_successful' not in df.columns:
    raise ValueError("Keine 'Status' und keine 'is_successful'-Spalte gefunden.")

# ------------------------------
# Feature-Auswahl
# ------------------------------
numerical_candidates = [
    'Fundingtotal', 'Fundingrounds', 'NumContributors', 'NumPartners', 
    'company_age_years', 'funding_duration_years', 'funding_rounds_per_year', 
    'funding_per_contributor', 'amount_milestones', 'milestones_per_year',
    'Fundingtotal_log', 'NumContributors_log', 'NumPartners_log', 
    'funding_per_contributor_log', 'avg_funding_per_round_log'
]
categorical_candidates = ['state_grouped', 'BusinessField_grouped']

available_columns = set(df.columns)
numerical_features = [f for f in numerical_candidates if f in available_columns]
categorical_features = [f for f in categorical_candidates if f in available_columns]

print("\nNumerische Features:", numerical_features)
print("Kategoriale Features:", categorical_features)

target = 'is_successful'
df_filtered = df.dropna(subset=[target])

# Falls zu wenige numerische Features erkannt werden:
if len(numerical_features) < 3:
    numerical_features = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    if target in numerical_features:
        numerical_features.remove(target)

# X und y definieren
X = df_filtered[numerical_features + categorical_features]
y = df_filtered[target].astype(int)

print(f"\nFeature-Set-Form: {X.shape}")
print(f"Zielvariable-Form: {y.shape}")

# ------------------------------
# Trainings-/Testsplit + SMOTE
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=52, stratify=y
)
print(f"\nTrainingsdaten: {X_train.shape[0]}")
print(f"Testdaten: {X_test.shape[0]}")
print(f"Verteilung im Training - Failed=0: {sum(y_train==0)}, Successful=1: {sum(y_train==1)}")

smote = SMOTE(random_state=52)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("\nNach SMOTE:")
print(f"Trainingssamples Failed=0: {sum(y_train_sm==0)}")
print(f"Trainingssamples Successful=1: {sum(y_train_sm==1)}")

# ------------------------------
# Preprocessing-Pipeline
# ------------------------------
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

transformers = [
    ('num', numeric_transformer, numerical_features)
]

if len(categorical_features) > 0:
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    transformers.append(('cat', categorical_transformer, categorical_features))

preprocessor = ColumnTransformer(transformers=transformers)

# ------------------------------
# Verschiedene Modelle definieren
# ------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        random_state=52, max_iter=2500, class_weight={0: 6, 1: 1}),
    "Random Forest": RandomForestClassifier(
        random_state=52, n_estimators=1500, class_weight={0: 6, 1: 1}),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=52, n_estimators=600, learning_rate=0.02),
    "Naive Bayes": GaussianNB(),
    "kNN": KNeighborsClassifier(), 
    "Neural Network": MLPClassifier(
        random_state=52, max_iter=10000, hidden_layer_sizes=(3000, 300, 100, 50, 20),
    )
}

# ------------------------------
# Training + Evaluation aller Modelle
# ------------------------------
results = []  # Hier sammeln wir die Ergebnisse

for model_name, clf in models.items():
    print(f"\n=== Training & Evaluation: {model_name} ===")
    # Pipeline mit Preprocessing + Modell
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    # Trainieren
    pipeline.fit(X_train_sm, y_train_sm)
    
    # Vorhersagen auf Testdaten
    y_pred = pipeline.predict(X_test)
    
    # Manche Modelle haben predict_proba, andere evtl. nicht
    if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
        y_pred_proba = pipeline.predict_proba(X_test)[:, 0]  # Wahrscheinlichkeit für Klasse "0"
        roc_auc = roc_auc_score(y_test, y_pred_proba)  # AUC für Klasse 0
    else:
        # z. B. bei manchen SVM-Varianten ohne probability=True
        y_pred_proba = None
        roc_auc = np.nan  # oder 0
    
    # Metriken (Fokus auf Klasse 0 = Failed)
    accuracy = (y_pred == y_test).mean()
    precision = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Konfusionsmatrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Ausgabe
    print(f"Confusion Matrix ({model_name}):")
    print(cm)
    print(f"AUC:  {roc_auc:.4f}")
    print(f"ACC:  {accuracy:.4f}")
    print(f"F1:   {f1:.4f}")
    print(f"Prec: {precision:.4f}")
    print(f"Rec:  {recall:.4f}")
    print(f"MCC:  {mcc:.4f}")
    
    # Speichern der Ergebnisse
    results.append({
        "Model": model_name,
        "AUC": roc_auc,
        "CA": accuracy,
        "F1": f1,
        "Prec": precision,
        "Recall": recall,
        "MCC": mcc
    })
    
    # Konfusionsmatrix plotten & speichern
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred: Failed', 'Pred: Success'],
                yticklabels=['True: Failed', 'True: Success'])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png"))
    plt.close()

# ------------------------------
# Ergebnisse tabellarisch ausgeben
# ------------------------------
results_df = pd.DataFrame(results)
print("\n=== Gesamtübersicht ===")
print(results_df)