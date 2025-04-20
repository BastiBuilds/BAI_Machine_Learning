import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, roc_curve,
                             precision_recall_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

# ------------------------------
# Pfaddefinitionen
# ------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
data_dir = os.path.join(project_dir, 'data')
results_dir = os.path.join(project_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

file_path = os.path.join(data_dir, 'my_vc_data_corrected.csv')
print(f"Lade Daten aus {file_path}...")

# ------------------------------
# Daten einlesen
# ------------------------------
try:
    df = pd.read_csv(file_path, sep=',', engine='python', on_bad_lines='warn')
    print(f"Datensatz-Form: {df.shape}")
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
# Zielvariable: Is_Successful (1 = Successful, 0 = Failed)
# ------------------------------
if 'Is_Successful' not in df.columns:
    raise ValueError("Keine 'Is_Successful'-Spalte gefunden.")
else:
    target = 'Is_Successful'
    df[target] = df[target].astype(int)

# ------------------------------
# Feature-Auswahl basierend auf neuer CSV
# ------------------------------
numerical_candidates = [
    'amount_milestones_corrected',
    'age_last_funding',
    'age_first_funding',
    'is_first_funding_the_biggest_funding',
    'First_funding_amount',
    'biggest_funding_amount_share_of_total_funding_%',
    'biggest_funding_amount',
    'Funding_rounds',
    'days_from_first_to_last_milestone',
    'average_days_per_milestone',
    'Funding_total',
    'funding_duration_days',
    'days_per_funding',
    'average_funding_size',
    'NumContributors',
    'NumPartners'
]
categorical_candidates = ['state', 'BusinessField']

available_columns = set(df.columns)
numerical_features   = [f for f in numerical_candidates   if f in available_columns]
categorical_features = [f for f in categorical_candidates if f in available_columns]

print("\nNumerische Features:", numerical_features)
print("Kategoriale Features:", categorical_features)

df_filtered = df.dropna(subset=[target])

# Alternative Feature-Auswahl, falls zu wenige numerische Features erkannt werden:
if len(numerical_features) < 3:
    numerical_features = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    if target in numerical_features:
        numerical_features.remove(target)

X = df_filtered[numerical_features + categorical_features]
y = df_filtered[target].astype(int)

print(f"\nFeature-Set-Form: {X.shape}")
print(f"Zielvariable-Form: {y.shape}")

# ---- Korrelationsanalyse ----
# Nutze die bestehenden numerical_features und die Zielvariable 'is_successful'
# Falls die Zielvariable noch nicht in numerical_features enthalten ist, fügen wir sie hinzu:
corr_cols = numerical_features.copy()
if target not in corr_cols:
    corr_cols.append(target)

# Berechne die Korrelationsmatrix
corr_matrix = df_filtered[corr_cols].corr()
print("\nKorrelationsmatrix:")
print(corr_matrix)

# Extrahiere die Korrelation der Features mit der Zielvariable (ohne die Zielvariable selbst)
target_corr = corr_matrix[target].drop(target)
# Sortiere nach absolutem Wert und wähle die Top 6
top6 = target_corr.abs().sort_values(ascending=False).head(6)
print("\nDie 6 wichtigsten Features bezüglich ihrer Korrelation mit der Zielvariable:")
print(top6)

# Optional: Aktualisiere die Feature-Auswahl anhand der Top 6 korrelierten Features.
selected_features = top6.index.tolist()
print("\nSelektierte Features für die Modellierung:")
print(selected_features)

numerical_features = selected_features

# ------------------------------
# Trainings-/Testsplit (SMOTE hier nicht mehr manuell)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=52, stratify=y
)
print(f"\nTrainingsdaten: {X_train.shape[0]}")
print(f"Testdaten: {X_test.shape[0]}")
print(f"Verteilung im Training - Failed=0: {sum(y_train==0)}, Successful=1: {sum(y_train==1)}")

# --- Bestimme k_neighbors für SMOTE so, dass es < minority class size ist ---
minority_count = y_train.value_counts().min()
smote_k = min(5, minority_count - 1) if minority_count > 1 else 1
print(f"Verwende SMOTE mit k_neighbors={smote_k}")

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
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    transformers.append(('cat', categorical_transformer, categorical_features))

preprocessor = ColumnTransformer(transformers=transformers)

# ------------------------------
# Modelle und Parametergrids definieren
# ------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        random_state=52, max_iter=2500, class_weight={0: 6, 1: 1}),
    "Random Forest": RandomForestClassifier(
        random_state=52, class_weight={0: 6, 1: 1}),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=52,),
    "Naive Bayes": GaussianNB(),
    "kNN": KNeighborsClassifier(), 
    "Neural Network": MLPClassifier(
        random_state=52, max_iter=10000)
}

param_grids = {
    "Logistic Regression": {
        'classifier__C': [0.01, 0.1, 1, 10, 100, 1000],
        'classifier__penalty': ['l2']
    },
    "Random Forest": {
        'classifier__n_estimators': [5000, 1000, 1500, 3000],
        'classifier__max_depth': [None, 20, 25, 30],
        'classifier__min_samples_split': [3, 5, 10, 20, 30]
    },
    "Gradient Boosting": {
        'classifier__n_estimators': [600, 1500, 2000, 4000],
        'classifier__learning_rate': [0.01, 0.02, 0.05],
        'classifier__max_depth': [3, 5, 9, 20]
    },
    "Neural Network": {
        'classifier__hidden_layer_sizes': [(5000, 300, 100, 50, 20), (2000, 500, 100)],
        'classifier__alpha': [0.00001, 0.001]

    }
}

# ------------------------------
# Training + Evaluation aller Modelle inkl. Hyperparameter-Tuning
# ------------------------------
results = []

for model_name, clf in models.items():
    print(f"\n=== Training & Evaluation: {model_name} ===")
    
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=52,
                        sampling_strategy=1.0,
                        k_neighbors=smote_k)),
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=10, random_state=52))),
        ('classifier',  clf)
    ])
    
    # GridSearch falls Parameter definiert sind:
    if model_name in param_grids:
        grid = GridSearchCV(pipeline,
                            param_grid=param_grids[model_name],
                            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=52),
                            scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_pipeline = grid.best_estimator_
        print(f"Beste Parameter für {model_name}:", grid.best_params_)
    else:
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline

    # -- Probability Calibration --
    # Falls das Modell predict_proba unterstützt, kalibriere es
    if hasattr(best_pipeline.named_steps['classifier'], "predict_proba"):
        # Verwende cv='prefit', da best_pipeline schon trainiert wurde.
        calibrated_clf = CalibratedClassifierCV(best_pipeline, method='sigmoid', cv='prefit')
        # Hier kann man als Basis den ursprünglichen Trainingssatz (ohne Oversampling) nehmen,
        # um Überanpassung zu verringern:
        calibrated_clf.fit(X_train, y_train)
        y_pred_proba = calibrated_clf.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = None

    # -- Threshold Optimierung --
    if y_pred_proba is not None:
        # F1-basierte Threshold-Optimierung (bestehende Methode)
        thresholds = np.linspace(0.0, 1.0, 101)
        f1_scores = []
        for thresh in thresholds:
            y_pred_thresh = (y_pred_proba >= thresh).astype(int)
            f1_scores.append(f1_score(y_test, y_pred_thresh, pos_label=1, zero_division=0))
        best_threshold_f1 = thresholds[np.argmax(f1_scores)]
        print(f"Optimale Schwelle (F1) für {model_name}: {best_threshold_f1:.2f}")
        
        # Zusätzlich: Optimierung mittels Youden-Index (als Proxy für AUC-Optimierung)
        best_threshold_youden = 0.5
        best_youden = -np.inf
        for thresh in thresholds:
            y_pred_thresh = (y_pred_proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            youden_index = sensitivity + specificity - 1
            if youden_index > best_youden:
                best_youden = youden_index
                best_threshold_youden = thresh
        print(f"Optimale Schwelle (Youden) für {model_name}: {best_threshold_youden:.2f}")
        
        # Hier entscheidest du, welchen Threshold du für die finale Vorhersage verwenden möchtest.
        # Z.B. Verwendung des F1-optimierten Thresholds:
        y_pred = (y_pred_proba >= best_threshold_f1).astype(int)
    else:
        y_pred = best_pipeline.predict(X_test)
    
    # Auswertung
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan
    accuracy = (y_pred == y_test).mean()
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix ({model_name}):")
    print(cm)
    print(f"AUC:  {roc_auc:.4f}")
    print(f"ACC:  {accuracy:.4f}")
    print(f"F1:   {f1:.4f}")
    print(f"Prec: {precision:.4f}")
    print(f"Rec:  {recall:.4f}")
    print(f"MCC:  {mcc:.4f}")
    
    results.append({
        "Model": model_name,
        "AUC": roc_auc,
        "ACC": accuracy,
        "F1": f1,
        "Prec": precision,
        "Recall": recall,
        "MCC": mcc
    })
    
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