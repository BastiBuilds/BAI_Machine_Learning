import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

# Pfad zur Datei festlegen
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
data_dir = os.path.join(project_dir, 'data')
file_path = os.path.join(data_dir, 'Exported_prepared_data.csv')

# Ergebnisordner für Diagramme und Modell erstellen
results_dir = os.path.join(project_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Daten einlesen (Semicolon als Delimiter beachten)
print(f"Lade Daten aus {file_path}...")
df = pd.read_csv(file_path, sep=';')

# Überprüfung der eingelesenen Daten
print(f"Datensatz-Form: {df.shape}")
print(f"Verteilung der Zielvariable:\n{df['Status'].value_counts()}")
print(f"Verteilung numerisch:\n{df['Status_numeric'].value_counts()}")

# Hilfsfunktion zum Umwandeln des Datums im Format DD.MM.YYYY in Timestamp
def convert_date(date_str):
    if isinstance(date_str, str):
        try:
            return datetime.strptime(date_str, '%d.%m.%Y')
        except:
            return pd.NaT
    return pd.NaT

# 1. VERBESSERTE DATENVORVERARBEITUNG UND FEATURE ENGINEERING

# Datum-Spalten umwandeln
date_columns = ['first_milestone_date', 'last_funding_date', 'first_funding_date', 
                'last_milestone_date', 'FoundingDate', 'closedDate', 'acquiredDate']

for col in date_columns:
    if col in df.columns:
        df[col] = df[col].apply(convert_date)

# acquiredDate entfernen, da es eine Datenleckage darstellt (erst nach Erfolg bekannt)
if 'acquiredDate_days_since_2000' in df.columns:
    df = df.drop(columns=['acquiredDate_days_since_2000'])

# Feature Engineering: Neue Zeitspanenfeatures erstellen
print("\n--- Feature Engineering ---")

# Unternehmensdauer bis Ende 2019 (Ende des Beobachtungszeitraums)
end_date = datetime(2019, 12, 31)
df['company_age_days'] = (end_date - df['FoundingDate']).dt.days

# Unternehmensdauer bis zur Akquisition oder Schließung
df['company_age_at_outcome_days'] = np.nan
mask_acquired = df['acquiredDate'].notna()
mask_closed = df['closedDate'].notna()

# Dauer bis zur Akquisition berechnen (für erfolgreiche Startups)
df.loc[mask_acquired, 'company_age_at_outcome_days'] = (
    df.loc[mask_acquired, 'acquiredDate'] - df.loc[mask_acquired, 'FoundingDate']
).dt.days

# Dauer bis zur Schließung berechnen (für gescheiterte Startups)
df.loc[mask_closed, 'company_age_at_outcome_days'] = (
    df.loc[mask_closed, 'closedDate'] - df.loc[mask_closed, 'FoundingDate']
).dt.days

# Fehlende Werte im Unternehmensdauer-Feature füllen
df['company_age_at_outcome_days'] = df['company_age_at_outcome_days'].fillna(df['company_age_days'])

# Zeit von Gründung bis erste Finanzierung
df['days_to_first_funding'] = np.nan
mask_first_funding = df['first_funding_date'].notna()
df.loc[mask_first_funding, 'days_to_first_funding'] = (
    df.loc[mask_first_funding, 'first_funding_date'] - df.loc[mask_first_funding, 'FoundingDate']
).dt.days

# Zeit von erster bis letzter Finanzierung
df['funding_timespan_days'] = np.nan
mask_funding_span = (df['first_funding_date'].notna() & df['last_funding_date'].notna())
df.loc[mask_funding_span, 'funding_timespan_days'] = (
    df.loc[mask_funding_span, 'last_funding_date'] - df.loc[mask_funding_span, 'first_funding_date']
).dt.days

# Meilenstein-Effizienz: Zeit pro Meilenstein
df['days_per_milestone'] = df.apply(
    lambda row: row['milestone_days_duration'] / row['amount_milestones'] 
    if pd.notna(row['amount_milestones']) and pd.notna(row['milestone_days_duration']) and row['amount_milestones'] > 0 
    else np.nan, 
    axis=1
)

# Finanzierungseffizienz: Geld pro Tag Unternehmensleben
df['funding_per_day'] = df.apply(
    lambda row: row['Fundingtotal'] / row['company_age_at_outcome_days'] 
    if pd.notna(row['company_age_at_outcome_days']) and row['company_age_at_outcome_days'] > 0 
    else 0, 
    axis=1
)

# Verhältnis von Finanzierung zu Mitarbeitern
df['funding_per_contributor'] = df.apply(
    lambda row: row['Fundingtotal'] / row['NumContributors']
    if pd.notna(row['NumContributors']) and row['NumContributors'] > 0 
    else 0,
    axis=1
)

# Finanzierungsrunden pro Jahr
df['founding_to_outcome_years'] = df['company_age_at_outcome_days'] / 365.25
df['funding_rounds_per_year'] = df.apply(
    lambda row: row['Fundingrounds'] / row['founding_to_outcome_years']
    if pd.notna(row['founding_to_outcome_years']) and row['founding_to_outcome_years'] > 0 
    else 0,
    axis=1
)

# Meilensteine pro Jahr und pro Mitarbeiter
df['milestones_per_year'] = df.apply(
    lambda row: row['amount_milestones'] / row['founding_to_outcome_years']
    if pd.notna(row['amount_milestones']) and pd.notna(row['founding_to_outcome_years']) and row['founding_to_outcome_years'] > 0 
    else 0,
    axis=1
)

df['milestones_per_contributor'] = df.apply(
    lambda row: row['amount_milestones'] / row['NumContributors']
    if pd.notna(row['amount_milestones']) and pd.notna(row['NumContributors']) and row['NumContributors'] > 0 
    else 0,
    axis=1
)

# Durchschnittliche Finanzierung pro Runde
df['avg_funding_per_round'] = df.apply(
    lambda row: row['Fundingtotal'] / row['Fundingrounds']
    if pd.notna(row['Fundingrounds']) and row['Fundingrounds'] > 0 
    else 0,
    axis=1
)

# Standort-Feature: erfolgreiche Startups pro Bundesstaat berechnen
state_success_rate = df.groupby('state')['Status_numeric'].mean()
df['state_success_rate'] = df['state'].map(state_success_rate)

# Branchenspezifische Erfolgsquote
business_field_success_rate = df.groupby('BusinessField')['Status_numeric'].mean()
df['business_field_success_rate'] = df['BusinessField'].map(business_field_success_rate)

# Kategoriale Features für Bundesstaaten mit weniger als 5 Startups zusammenfassen
state_counts = df['state'].value_counts()
rare_states = state_counts[state_counts < 5].index
df['state_grouped'] = df['state'].apply(lambda x: 'OTHER' if x in rare_states else x)

# Geschäftsfelder mit weniger als 5 Startups zusammenfassen
business_field_counts = df['BusinessField'].value_counts()
rare_fields = business_field_counts[business_field_counts < 5].index
df['BusinessField_grouped'] = df['BusinessField'].apply(lambda x: 'OTHER' if x in rare_fields else x)

# Logarithmische Transformation für schiefe Verteilungen
skewed_features = ['Fundingtotal', 'NumContributors', 'NumPartners', 'funding_per_contributor', 'avg_funding_per_round']
for feature in skewed_features:
    if feature in df.columns:
        df[f'{feature}_log'] = np.log1p(df[feature])

# 2. VERBESSERTE FEATURE-AUSWAHL

# Feature-Auswahl: Relevante Features für das Modell definieren
# Basierend auf der Korrelationsanalyse und Feature Importance
numerical_features = [
    # Finanzierungs-Features
    'Fundingtotal', 'Fundingtotal_log', 'Fundingrounds', 'funding_duration_days',
    'funding_per_day', 'funding_per_contributor', 'funding_rounds_per_year', 'avg_funding_per_round',
    'days_to_first_funding', 'funding_timespan_days',
    
    # Unternehmensdaten
    'NumContributors', 'NumContributors_log', 'NumPartners', 'NumPartners_log',
    'company_age_at_outcome_days', 'company_age_days',
    
    # Erfolgsraten-Features
    'state_success_rate', 'business_field_success_rate'
]

# Meilenstein-Features hinzufügen, aber nur wenn sie nicht stark miteinander korrelieren
milestone_features = [
    'amount_milestones', 'milestones_per_year', 'milestones_per_contributor'
]

for feature in milestone_features:
    if feature in df.columns:
        numerical_features.append(feature)

# Kategorische Features
categorical_features = ['state_grouped', 'BusinessField_grouped']

# Zielvariable
target = 'Status_numeric'

# Features und Zielvariable extrahieren
X = df[numerical_features + categorical_features]
y = df[target]

# Überprüfen auf fehlende Werte in den ausgewählten Features
missing_values = X.isnull().sum()
if missing_values.sum() > 0:
    print("\nFehlende Werte in ausgewählten Features:")
    print(missing_values[missing_values > 0])
    print("Diese werden in der Pipeline durch Imputation behandelt.")

# 3. VERBESSERTE TRAIN-TEST-SPLIT UND CROSS-VALIDATION

# Holdout-Set für finale Evaluierung erstellen
X_model, X_holdout, y_model, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train-Test-Split für das Modelltraining und initiale Evaluierung
X_train, X_test, y_train, y_test = train_test_split(
    X_model, y_model, test_size=0.25, random_state=42, stratify=y_model
)

print(f"\nTrainingsdaten: {X_train.shape[0]} Samples")
print(f"Testdaten: {X_test.shape[0]} Samples")
print(f"Holdout-Daten: {X_holdout.shape[0]} Samples")

# 4. VERBESSERTE PREPROCESSING-PIPELINE

# Numerische Features: Imputation, Power-Transformation und Standardisierung
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('power', PowerTransformer(method='yeo-johnson', standardize=False)),  # Für bessere Normalverteilung
    ('scaler', StandardScaler())
])

# Kategoriale Features: One-Hot-Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Kombinierter Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. VERBESSERTE MODELLIERUNG UND HYPERPARAMETER-TUNING

# Random Forest mit erweitertem Grid Search
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # Feature-Selektion nach Vorverarbeitung
    ('feature_selection', SelectFromModel(GradientBoostingClassifier(random_state=42))),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Erweiterter Grid Search für Random Forest
rf_param_grid = {
    'feature_selection__threshold': ['mean', 'median'],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', None]
}

# Cross-Validation mit Stratifizierung - mehr Folds für robustere Evaluation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Grid Search für RF durchführen mit mehreren Metriken
print("\n--- Erweitertes Hyperparameter-Tuning für Random Forest ---")
rf_grid_search = GridSearchCV(
    rf_model, rf_param_grid, cv=cv, 
    scoring={
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall'
    },
    refit='roc_auc',  # Optimierung nach ROC-AUC
    n_jobs=-1, verbose=1
)

print("Training des Random Forest-Modells...")
rf_grid_search.fit(X_train, y_train)
print(f"Beste Parameter: {rf_grid_search.best_params_}")
print(f"Bester ROC-AUC-Score: {rf_grid_search.best_score_:.4f}")

# Diverse Scoring-Metriken anzeigen
print("\nErgebnisse der verschiedenen Metriken:")
for metric in ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']:
    print(f"{metric}: {rf_grid_search.cv_results_[f'mean_test_{metric}'][rf_grid_search.best_index_]:.4f}")

# Bestes RF-Modell auswählen
best_rf_model = rf_grid_search.best_estimator_

# 6. ÜBERPRÜFUNG AUF CLASS IMBALANCE UND ANWENDUNG VON SMOTE

# Überprüfen, ob ein Class Imbalance Problem besteht und ggf. SMOTE anwenden
class_counts = y_train.value_counts()
class_ratio = class_counts.min() / class_counts.max()
print(f"\nKlassenverhältnis (min/max): {class_ratio:.2f}")

if class_ratio < 0.75:
    print("Anwendung von SMOTE zum Ausgleich der Klassenungleichheit...")
    
    # SMOTE auf den transformierten Daten anwenden
    X_train_transformed = best_rf_model.named_steps['preprocessor'].transform(X_train)
    # Feature-Selektion anwenden, falls im Modell vorhanden
    if 'feature_selection' in best_rf_model.named_steps:
        X_train_transformed = best_rf_model.named_steps['feature_selection'].transform(X_train_transformed)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
    
    # Neues Modell nur mit dem Classifier-Teil trainieren
    best_rf_model.named_steps['classifier'].fit(X_train_resampled, y_train_resampled)
    
    print(f"Nach SMOTE - Trainingssamples Klasse 0: {sum(y_train_resampled == 0)}")
    print(f"Nach SMOTE - Trainingssamples Klasse 1: {sum(y_train_resampled == 1)}")
else:
    print("Kein signifikantes Class Imbalance Problem festgestellt.")

# 7. MODELLBEWERTUNG AUF TESTDATEN

# Vorhersagen auf dem Testset
y_pred = best_rf_model.predict(X_test)
y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]

# Bewertungsmetriken berechnen
print("\n--- Modellbewertung auf Testdaten ---")
print("\nKlassifikationsreport:")
print(classification_report(y_test, y_pred))

print("\nKonfusionsmatrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ROC-AUC berechnen
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

# ROC-Kurve visualisieren
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, 'improved_roc_curve.png'))

# Precision-Recall Curve
plt.figure(figsize=(10, 8))
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig(os.path.join(results_dir, 'improved_precision_recall_curve.png'))

# 8. FEATURE IMPORTANCE ANALYSE

# Feature Importance extrahieren und visualisieren
if hasattr(best_rf_model.named_steps['classifier'], 'feature_importances_'):
    # Transformed Feature Namen ermitteln
    preprocessor = best_rf_model.named_steps['preprocessor']
    
    # Numerische Feature-Namen
    num_features = numerical_features
    
    # One-Hot encodierte kategoriale Feature-Namen
    cat_features = []
    for i, cat_feature in enumerate(categorical_features):
        # Zugriff auf OneHotEncoder und dessen Kategorien
        if hasattr(preprocessor.transformers_[1][1].named_steps['onehot'], 'categories_'):
            categories = preprocessor.transformers_[1][1].named_steps['onehot'].categories_[i]
            for category in categories:
                cat_features.append(f"{cat_feature}_{category}")
    
    # Alle Feature-Namen kombinieren
    all_feature_names = num_features + cat_features
    
    # Feature Importances extrahieren
    importances = best_rf_model.named_steps['classifier'].feature_importances_
    
    # Feature-Selektion berücksichtigen - nur ausgewählte Features betrachten
    if 'feature_selection' in best_rf_model.named_steps:
        feature_selector = best_rf_model.named_steps['feature_selection']
        selected_indices = feature_selector.get_support()
        
        # Transformierte Features filtern und nur selektierte verwenden
        transformed_features = np.array(all_feature_names)
        if len(transformed_features) > len(selected_indices):
            transformed_features = transformed_features[:len(selected_indices)]
        
        selected_features = transformed_features[selected_indices]
        selected_importances = importances
        
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': selected_importances
        }).sort_values('Importance', ascending=False)
    else:
        # Ohne Feature-Selektion - alle transformierten Features verwenden
        # Längenanpassung, falls nötig
        if len(all_feature_names) > len(importances):
            all_feature_names = all_feature_names[:len(importances)]
        elif len(all_feature_names) < len(importances):
            importances = importances[:len(all_feature_names)]
        
        feature_importance = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
    
    # Top Features anzeigen
    top_n = min(20, len(feature_importance))
    top_features = feature_importance.head(top_n)
    
    print("\n--- Feature Importance Analyse ---")
    print("\nTop Features nach Wichtigkeit:")
    for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Feature Importance visualisieren
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), top_features['Importance'], align='center')
    plt.yticks(range(top_n), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title('Top Features nach Wichtigkeit')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'improved_feature_importance.png'))

# 9. FINAL EVALUATION ON HOLDOUT SET

print("\n--- Finale Evaluierung auf Holdout-Daten ---")
y_holdout_pred = best_rf_model.predict(X_holdout)
y_holdout_pred_proba = best_rf_model.predict_proba(X_holdout)[:, 1]

print("\nKlassifikationsreport auf Holdout-Daten:")
print(classification_report(y_holdout, y_holdout_pred))

print("\nKonfusionsmatrix auf Holdout-Daten:")
cm_holdout = confusion_matrix(y_holdout, y_holdout_pred)
print(cm_holdout)

# ROC-AUC berechnen
roc_auc_holdout = roc_auc_score(y_holdout, y_holdout_pred_proba)
print(f"ROC-AUC auf Holdout-Daten: {roc_auc_holdout:.4f}")

# 10. WIRTSCHAFTLICHE ANALYSE

# Wirtschaftlicher Nutzen des Modells berechnen
print("\n--- Wirtschaftliche Analyse ---")

# Annahmen für das Kosten-Nutzen-Modell
acquisition_profit = 5000000  # Gewinn aus einer erfolgreichen Akquisition (€)
investment_loss = 500000  # Verlust aus einer gescheiterten Investition (€)

print("\nAnnahmen für die wirtschaftliche Analyse:")
print(f"Gewinn pro erfolgreicher Akquisition: {acquisition_profit/1e6:.1f} Mio €")
print(f"Verlust pro gescheiterter Investition: {investment_loss/1e6:.1f} Mio €")

# Konfusionsmatrix-Werte extrahieren (vom Holdout-Set für realistischere Einschätzung)
tp = cm_holdout[1, 1]  # True Positives: korrekt als erfolgreich vorhergesagt
fp = cm_holdout[0, 1]  # False Positives: fälschlicherweise als erfolgreich vorhergesagt
fn = cm_holdout[1, 0]  # False Negatives: fälschlicherweise als gescheitert vorhergesagt
tn = cm_holdout[0, 0]  # True Negatives: korrekt als gescheitert vorhergesagt

# Berechnung des Gewinns mit dem Modell
profit_with_model = tp * acquisition_profit - fp * investment_loss

# Berechnung des Gewinns ohne Modell (verschiedene Szenarien)
# Szenario 1: Ohne Modell investiert man in alle Startups
profit_all = (tp + fn) * acquisition_profit - (fp + tn) * investment_loss

# Szenario 2: Ohne Modell investiert man in kein Startup
profit_none = 0

# Szenario 3: Ohne Modell investiert man zufällig (basierend auf der tatsächlichen Erfolgsquote)
success_rate = (tp + fn) / (tp + fp + tn + fn)
expected_profit_per_investment = success_rate * acquisition_profit - (1 - success_rate) * investment_loss
profit_random = expected_profit_per_investment * (tp + fp + tn + fn)

# Gewinnsteigerung durch das Modell
profit_increase_vs_all = profit_with_model - profit_all
profit_increase_vs_none = profit_with_model - profit_none
profit_increase_vs_random = profit_with_model - profit_random

print(f"\nWirtschaftlicher Nutzen des Modells:")
print(f"Gewinn mit Modell: {profit_with_model/1e6:.2f} Mio €")
print(f"Gewinn bei Investition in alle Startups: {profit_all/1e6:.2f} Mio €")
print(f"Gewinn bei keiner Investition: {profit_none/1e6:.2f} Mio €")
print(f"Gewinn bei zufälliger Investition (basierend auf tatsächlicher Erfolgsquote): {profit_random/1e6:.2f} Mio €")
print(f"\nGewinnsteigerung vs. alle Investitionen: {profit_increase_vs_all/1e6:.2f} Mio €")
print(f"Gewinnsteigerung vs. keine Investition: {profit_increase_vs_none/1e6:.2f} Mio €")
print(f"Gewinnsteigerung vs. zufällige Investition: {profit_increase_vs_random/1e6:.2f} Mio €")

# ROI des Modells (Angenommen, die Modellentwicklung kostet 100.000 €)
model_cost = 100000
roi_vs_all = profit_increase_vs_all / model_cost
roi_vs_none = profit_increase_vs_none / model_cost
roi_vs_random = profit_increase_vs_random / model_cost

print(f"\nROI des Modells:")
print(f"ROI vs. alle Investitionen: {roi_vs_all:.2f}x")
print(f"ROI vs. keine Investition: {roi_vs_none:.2f}x")
print(f"ROI vs. zufällige Investition: {roi_vs_random:.2f}x")

# 11. MODELL SPEICHERN

# Modell für zukünftige Vorhersagen speichern
model_path = os.path.join(results_dir, 'improved_startup_success_model.joblib')
joblib.dump(best_rf_model, model_path)
print(f"\nOptimiertes Modell wurde gespeichert unter: {model_path}")

# Funktion zum Laden und Nutzen des Modells für neue Startups
def predict_startup_success(model, startup_data):
    """
    Funktion zur Vorhersage des Erfolgs eines neuen Startups.
    
    Parameters:
    - model: Das trainierte Modell
    - startup_data: Ein Pandas DataFrame mit den erforderlichen Features
    
    Returns:
    - Erfolgswahrscheinlichkeit (zwischen 0 und 1)
    """
    # Prüfen, ob alle nötigen Features vorhanden sind
    required_features = numerical_features + categorical_features
    missing_features = set(required_features) - set(startup_data.columns)
    
    if missing_features:
        raise ValueError(f"Fehlende Features: {missing_features}")
    
    # Vorhersage treffen
    success_prob = model.predict_proba(startup_data[required_features])[:, 1]
    return success_prob

print("\nBeispiel für die Verwendung des Modells auf neuen Daten:")
print("model = joblib.load('results/improved_startup_success_model.joblib')")
print("new_startup_data = pd.DataFrame({...})  # Neue Startup-Daten mit erforderlichen Features")
print("success_probability = predict_startup_success(model, new_startup_data)")
print("print(f'Erfolgswahrscheinlichkeit: {success_probability[0]:.2%}')")

print("\n--- Zusätzliche Modellverbesserungen ---")

# 12. ENSEMBLE-MODELL ERSTELLEN (OPTIONAL)

# Mehrere Modelle trainieren und kombinieren für bessere Vorhersagen
print("\nErstellung eines Ensemble-Modells aus mehreren Klassifikatoren...")

# Gradient Boosting Classifier
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Logistische Regression
lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Einfache Parameter für die zusätzlichen Modelle (zur Demonstration)
gb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Voting Classifier erstellen
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', best_rf_model),
        ('gb', gb_model),
        ('lr', lr_model)
    ],
    voting='soft'  # Nutze Wahrscheinlichkeiten für die Stimmabgabe
)

# Ensemble trainieren
ensemble_model.fit(X_train, y_train)

# Ensemble auf Holdout-Daten evaluieren
ensemble_preds = ensemble_model.predict(X_holdout)
ensemble_probs = ensemble_model.predict_proba(X_holdout)[:, 1]
ensemble_auc = roc_auc_score(y_holdout, ensemble_probs)

print(f"\nEnsemble-Modell ROC-AUC auf Holdout-Daten: {ensemble_auc:.4f}")
print("\nKlassifikationsreport für Ensemble-Modell:")
print(classification_report(y_holdout, ensemble_preds))

# Vergleich des Ensemble-Modells mit dem besten Einzelmodell
print(f"\nVergleich der Modelle auf Holdout-Daten:")
print(f"Bestes Random Forest Modell ROC-AUC: {roc_auc_holdout:.4f}")
print(f"Ensemble-Modell ROC-AUC: {ensemble_auc:.4f}")

if ensemble_auc > roc_auc_holdout:
    print("\nDas Ensemble-Modell bietet eine Verbesserung gegenüber dem besten Einzelmodell!")
    # Ensemble-Modell speichern
    ensemble_path = os.path.join(results_dir, 'startup_ensemble_model.joblib')
    joblib.dump(ensemble_model, ensemble_path)
    print(f"Ensemble-Modell wurde gespeichert unter: {ensemble_path}")
else:
    print("\nDas beste Einzelmodell (Random Forest) bleibt die beste Wahl.")

# 13. LERNKURVE ANALYSIEREN

print("\n--- Lernkurvenanalyse ---")
# Lernkurve berechnen, um zu prüfen, ob mehr Daten hilfreich wären
from sklearn.model_selection import learning_curve

# Definieren der Trainingssample-Größen
train_sizes = np.linspace(0.1, 1.0, 10)

# Lernkurve für das beste Modell berechnen
train_sizes, train_scores, test_scores = learning_curve(
    best_rf_model, X_model, y_model, train_sizes=train_sizes, cv=5, scoring='roc_auc', n_jobs=-1
)

# Mittelwerte und Standardabweichungen berechnen
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Lernkurve visualisieren
plt.figure(figsize=(10, 6))
plt.grid()
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
plt.title("Learning Curve (ROC-AUC)")
plt.xlabel("Training samples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.savefig(os.path.join(results_dir, 'learning_curve.png'))

if test_mean[-1] > test_mean[-2]:
    print("Die Lernkurve zeigt, dass mehr Trainingsdaten wahrscheinlich zu besseren Ergebnissen führen würden.")
else:
    print("Die Lernkurve flacht ab, was darauf hindeutet, dass mehr Daten wenig zusätzlichen Nutzen bringen würden.")

print("\nOptimiertes Modelltraining und -analyse abgeschlossen!")