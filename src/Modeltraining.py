import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Pfaddefinitionen
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
data_dir = os.path.join(project_dir, 'data')
file_path = os.path.join(data_dir, 'Exported_prepared_data_cleaned.csv')
results_dir = os.path.join(project_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Daten einlesen
print(f"Lade Daten aus {file_path}...")
df = pd.read_csv(file_path, sep=';')
print(f"Datensatz-Form: {df.shape}")

# Datumsspalten vektorisiert umwandeln
date_columns = ['first_milestone_date', 'last_funding_date', 'first_funding_date', 
                'last_milestone_date', 'FoundingDate', 'closedDate', 'acquiredDate']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='%d.%m.%Y', errors='coerce')

# Datenleckage-Spalte entfernen
if 'acquiredDate_days_since_2000' in df.columns:
    df.drop(columns=['acquiredDate_days_since_2000'], inplace=True)

print("\n--- Feature Engineering ---")
end_date = datetime(2019, 12, 31)
df['company_age_days'] = (end_date - df['FoundingDate']).dt.days

# Unternehmensalter zum Outcome berechnen (vektorisiert)
df['company_age_at_outcome_days'] = np.where(df['acquiredDate'].notna(), 
                                             (df['acquiredDate'] - df['FoundingDate']).dt.days, np.nan)
df['company_age_at_outcome_days'] = np.where(df['closedDate'].notna(), 
                                             (df['closedDate'] - df['FoundingDate']).dt.days, 
                                             df['company_age_at_outcome_days'])
df['company_age_at_outcome_days'] = df['company_age_at_outcome_days'].fillna(df['company_age_days'])

# Weitere Features vektorisiert berechnen
df['days_to_first_funding'] = np.where(df['first_funding_date'].notna(),
                                       (df['first_funding_date'] - df['FoundingDate']).dt.days, np.nan)
df['funding_timespan_days'] = np.where(df['first_funding_date'].notna() & df['last_funding_date'].notna(),
                                       (df['last_funding_date'] - df['first_funding_date']).dt.days, np.nan)
df['days_per_milestone'] = np.where((df['amount_milestones'] > 0) & df['milestone_days_duration'].notna(),
                                    df['milestone_days_duration'] / df['amount_milestones'], np.nan)
df['funding_per_day'] = np.where((df['company_age_at_outcome_days'] > 0) & df['company_age_at_outcome_days'].notna(),
                                 df['Fundingtotal'] / df['company_age_at_outcome_days'], 0)
df['funding_per_contributor'] = np.where((df['NumContributors'] > 0) & df['NumContributors'].notna(),
                                         df['Fundingtotal'] / df['NumContributors'], 0)
df['founding_to_outcome_years'] = df['company_age_at_outcome_days'] / 365.25
df['funding_rounds_per_year'] = np.where((df['founding_to_outcome_years'] > 0) & df['founding_to_outcome_years'].notna(),
                                          df['Fundingrounds'] / df['founding_to_outcome_years'], 0)
df['milestones_per_year'] = np.where((df['amount_milestones'].notna()) & (df['founding_to_outcome_years'] > 0),
                                      df['amount_milestones'] / df['founding_to_outcome_years'], 0)
df['milestones_per_contributor'] = np.where((df['NumContributors'] > 0) & df['NumContributors'].notna(),
                                            df['amount_milestones'] / df['NumContributors'], 0)
df['avg_funding_per_round'] = np.where((df['Fundingrounds'] > 0) & df['Fundingrounds'].notna(),
                                       df['Fundingtotal'] / df['Fundingrounds'], 0)

# Erfolgsquoten anhand von State und BusinessField ermitteln
state_success_rate = df.groupby('state')['Status_numeric'].mean()
df['state_success_rate'] = df['state'].map(state_success_rate)
business_field_success_rate = df.groupby('BusinessField')['Status_numeric'].mean()
df['business_field_success_rate'] = df['BusinessField'].map(business_field_success_rate)

# Seltene States und BusinessFields gruppieren
state_counts = df['state'].value_counts()
rare_states = state_counts[state_counts < 5].index
df['state_grouped'] = df['state'].where(~df['state'].isin(rare_states), 'OTHER')
business_field_counts = df['BusinessField'].value_counts()
rare_fields = business_field_counts[business_field_counts < 5].index
df['BusinessField_grouped'] = df['BusinessField'].where(~df['BusinessField'].isin(rare_fields), 'OTHER')

# Logarithmische Transformation schiefer Features (vektorisiert)
skewed_features = ['Fundingtotal', 'NumContributors', 'NumPartners', 'funding_per_contributor', 'avg_funding_per_round']
for feature in skewed_features:
    if feature in df.columns:
        df[f'{feature}_log'] = np.log1p(df[feature])

# Features und Zielvariable definieren
numerical_features = [
    'Fundingtotal', 'Fundingtotal_log', 'Fundingrounds', 'funding_duration_days',
    'funding_per_day', 'funding_per_contributor', 'funding_rounds_per_year', 'avg_funding_per_round',
    'days_to_first_funding', 'funding_timespan_days',
    'NumContributors', 'NumContributors_log', 'NumPartners', 'NumPartners_log',
    'company_age_at_outcome_days', 'company_age_days',
    'state_success_rate', 'business_field_success_rate'
]
milestone_features = ['amount_milestones', 'milestones_per_year', 'milestones_per_contributor']
for feature in milestone_features:
    if feature in df.columns:
        numerical_features.append(feature)
categorical_features = ['state_grouped', 'BusinessField_grouped']
target = 'Status_numeric'

X = df[numerical_features + categorical_features]
y = df[target]

# Fehlende Werte prüfen (werden in der Pipeline imputiert)
missing_values = X.isnull().sum()
if missing_values.sum() > 0:
    print("\nFehlende Werte in ausgewählten Features:")
    print(missing_values[missing_values > 0])
    print("Diese werden in der Pipeline durch Imputation behandelt.")

# Datenaufteilung
X_model, X_holdout, y_model, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.25, random_state=42, stratify=y_model)
print(f"\nTrainingsdaten: {X_train.shape[0]} Samples")
print(f"Testdaten: {X_test.shape[0]} Samples")
print(f"Holdout-Daten: {X_holdout.shape[0]} Samples")

# Preprocessing-Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('power', PowerTransformer(method='yeo-johnson', standardize=False)),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Modellpipeline und reduzierter Grid Search (schneller)
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectFromModel(GradientBoostingClassifier(random_state=42))),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])
rf_param_grid = {
    'feature_selection__threshold': ['mean'],       # 1 Option statt 2
    'classifier__n_estimators': [100, 200],           # weniger Bäume
    'classifier__max_depth': [None, 10],              # weniger Optionen
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__max_features': ['sqrt']
}
# Für schnellere Ausführung werden 5 CV-Folds statt 10 verwendet
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Hyperparameter-Tuning für Random Forest ---")
rf_grid_search = GridSearchCV(
    rf_model, rf_param_grid, cv=cv, scoring='roc_auc', refit=True, n_jobs=-1, verbose=1
)
print("Training des Random Forest-Modells...")
rf_grid_search.fit(X_train, y_train)
print(f"Beste Parameter: {rf_grid_search.best_params_}")
print(f"Bester ROC-AUC-Score: {rf_grid_search.best_score_:.4f}")

print("\nErgebnisse der verschiedenen Metriken:")
for metric in ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']:
    if f"mean_test_{metric}" in rf_grid_search.cv_results_:
        print(f"{metric}: {rf_grid_search.cv_results_[f'mean_test_{metric}'][rf_grid_search.best_index_]:.4f}")

best_rf_model = rf_grid_search.best_estimator_

# Überprüfung von Class Imbalance und ggf. Anwendung von SMOTE
class_counts = y_train.value_counts()
class_ratio = class_counts.min() / class_counts.max()
print(f"\nKlassenverhältnis (min/max): {class_ratio:.2f}")
if class_ratio < 0.75:
    print("Anwendung von SMOTE...")
    X_train_transformed = best_rf_model.named_steps['preprocessor'].transform(X_train)
    if 'feature_selection' in best_rf_model.named_steps:
        X_train_transformed = best_rf_model.named_steps['feature_selection'].transform(X_train_transformed)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
    best_rf_model.named_steps['classifier'].fit(X_train_resampled, y_train_resampled)
    print(f"Nach SMOTE - Trainingssamples Klasse 0: {sum(y_train_resampled == 0)}")
    print(f"Nach SMOTE - Trainingssamples Klasse 1: {sum(y_train_resampled == 1)}")
else:
    print("Kein signifikantes Class Imbalance Problem festgestellt.")

# Evaluation auf dem Testset
y_pred = best_rf_model.predict(X_test)
y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]
print("\n--- Modellbewertung auf Testdaten ---")
print("\nKlassifikationsreport:")
print(classification_report(y_test, y_pred))
print("\nKonfusionsmatrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, 'optimized_roc_curve.png'))

plt.figure(figsize=(10, 8))
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig(os.path.join(results_dir, 'optimized_precision_recall_curve.png'))

# Feature Importance Analyse (wenn verfügbar)
if hasattr(best_rf_model.named_steps['classifier'], 'feature_importances_'):
    preprocessor = best_rf_model.named_steps['preprocessor']
    num_features = numerical_features
    cat_features = []
    for i, cat_feature in enumerate(categorical_features):
        if hasattr(preprocessor.transformers_[1][1].named_steps['onehot'], 'categories_'):
            categories = preprocessor.transformers_[1][1].named_steps['onehot'].categories_[i]
            cat_features.extend([f"{cat_feature}_{cat}" for cat in categories])
    all_feature_names = num_features + cat_features
    importances = best_rf_model.named_steps['classifier'].feature_importances_
    if 'feature_selection' in best_rf_model.named_steps:
        feature_selector = best_rf_model.named_steps['feature_selection']
        selected_indices = feature_selector.get_support()
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
        if len(all_feature_names) > len(importances):
            all_feature_names = all_feature_names[:len(importances)]
        elif len(all_feature_names) < len(importances):
            importances = importances[:len(all_feature_names)]
        feature_importance = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
    top_n = min(20, len(feature_importance))
    top_features = feature_importance.head(top_n)
    print("\n--- Feature Importance Analyse ---")
    print("\nTop Features:")
    for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
        print(f"{i+1}. {feature}: {importance:.4f}")
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), top_features['Importance'], align='center')
    plt.yticks(range(top_n), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title('Top Features')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'optimized_feature_importance.png'))

print("\n--- Finale Evaluierung auf Holdout-Daten ---")
y_holdout_pred = best_rf_model.predict(X_holdout)
y_holdout_pred_proba = best_rf_model.predict_proba(X_holdout)[:, 1]
print("\nKlassifikationsreport auf Holdout-Daten:")
print(classification_report(y_holdout, y_holdout_pred))
print("\nKonfusionsmatrix auf Holdout-Daten:")
cm_holdout = confusion_matrix(y_holdout, y_holdout_pred)
print(cm_holdout)
roc_auc_holdout = roc_auc_score(y_holdout, y_holdout_pred_proba)
print(f"ROC-AUC auf Holdout-Daten: {roc_auc_holdout:.4f}")

# Wirtschaftliche Analyse
acquisition_profit = 5000000  # Gewinn aus erfolgreicher Akquisition (€)
investment_loss = 500000      # Verlust aus gescheiterter Investition (€)
print("\n--- Wirtschaftliche Analyse ---")
print(f"Gewinn pro erfolgreicher Akquisition: {acquisition_profit/1e6:.1f} Mio €")
print(f"Verlust pro gescheiterter Investition: {investment_loss/1e6:.1f} Mio €")
tp = cm_holdout[1, 1]
fp = cm_holdout[0, 1]
fn = cm_holdout[1, 0]
tn = cm_holdout[0, 0]
profit_with_model = tp * acquisition_profit - fp * investment_loss
profit_all = (tp + fn) * acquisition_profit - (fp + tn) * investment_loss
profit_none = 0
success_rate = (tp + fn) / (tp + fp + tn + fn)
expected_profit_per_investment = success_rate * acquisition_profit - (1 - success_rate) * investment_loss
profit_random = expected_profit_per_investment * (tp + fp + tn + fn)
profit_increase_vs_all = profit_with_model - profit_all
profit_increase_vs_none = profit_with_model - profit_none
profit_increase_vs_random = profit_with_model - profit_random
print(f"\nGewinn mit Modell: {profit_with_model/1e6:.2f} Mio €")
print(f"Gewinn bei Investition in alle Startups: {profit_all/1e6:.2f} Mio €")
print(f"Gewinn bei keiner Investition: {profit_none/1e6:.2f} Mio €")
print(f"Gewinn bei zufälliger Investition: {profit_random/1e6:.2f} Mio €")
print(f"\nGewinnsteigerung vs. alle Investitionen: {profit_increase_vs_all/1e6:.2f} Mio €")
print(f"Gewinnsteigerung vs. keine Investition: {profit_increase_vs_none/1e6:.2f} Mio €")
print(f"Gewinnsteigerung vs. zufällige Investition: {profit_increase_vs_random/1e6:.2f} Mio €")
model_cost = 100000
roi_vs_all = profit_increase_vs_all / model_cost
roi_vs_none = profit_increase_vs_none / model_cost
roi_vs_random = profit_increase_vs_random / model_cost
print(f"\nROI des Modells:")
print(f"ROI vs. alle Investitionen: {roi_vs_all:.2f}x")
print(f"ROI vs. keine Investition: {roi_vs_none:.2f}x")
print(f"ROI vs. zufällige Investition: {roi_vs_random:.2f}x")

# Modell speichern
model_path = os.path.join(results_dir, 'optimized_startup_success_model.joblib')
joblib.dump(best_rf_model, model_path)
print(f"\nOptimiertes Modell wurde gespeichert unter: {model_path}")

def predict_startup_success(model, startup_data):
    """
    Vorhersage des Erfolgs eines neuen Startups.
    """
    required_features = numerical_features + categorical_features
    missing_features = set(required_features) - set(startup_data.columns)
    if missing_features:
        raise ValueError(f"Fehlende Features: {missing_features}")
    success_prob = model.predict_proba(startup_data[required_features])[:, 1]
    return success_prob

print("\nBeispiel für die Verwendung des Modells:")
print("model = joblib.load('results/optimized_startup_success_model.joblib')")
print("new_startup_data = pd.DataFrame({...})  # Neue Startup-Daten")
print("success_probability = predict_startup_success(model, new_startup_data)")
print("print(f'Erfolgswahrscheinlichkeit: {success_probability[0]:.2%}')")

# Optional: Ensemble-Modell erstellen
print("\n--- Ensemble-Modell (optional) ---")
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])
lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
gb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
ensemble_model = VotingClassifier(estimators=[
    ('rf', best_rf_model),
    ('gb', gb_model),
    ('lr', lr_model)
], voting='soft')
ensemble_model.fit(X_train, y_train)
ensemble_preds = ensemble_model.predict(X_holdout)
ensemble_probs = ensemble_model.predict_proba(X_holdout)[:, 1]
ensemble_auc = roc_auc_score(y_holdout, ensemble_probs)
print(f"\nEnsemble-Modell ROC-AUC auf Holdout-Daten: {ensemble_auc:.4f}")
print("\nKlassifikationsreport für Ensemble-Modell:")
print(classification_report(y_holdout, ensemble_preds))
print(f"\nVergleich der Modelle auf Holdout-Daten:")
print(f"Bestes Random Forest Modell ROC-AUC: {roc_auc_holdout:.4f}")
print(f"Ensemble-Modell ROC-AUC: {ensemble_auc:.4f}")
if ensemble_auc > roc_auc_holdout:
    print("\nEnsemble-Modell bietet eine Verbesserung!")
    ensemble_path = os.path.join(results_dir, 'startup_ensemble_model.joblib')
    joblib.dump(ensemble_model, ensemble_path)
    print(f"Ensemble-Modell wurde gespeichert unter: {ensemble_path}")
else:
    print("\nDas beste Einzelmodell bleibt die beste Wahl.")

# Lernkurvenanalyse (mit reduzierten Trainingsgrößen für schnellere Berechnung)
print("\n--- Lernkurvenanalyse ---")
train_sizes, train_scores, test_scores = learning_curve(
    best_rf_model, X_model, y_model, train_sizes=np.linspace(0.1, 1.0, 5), cv=5, scoring='roc_auc', n_jobs=-1
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
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
plt.savefig(os.path.join(results_dir, 'optimized_learning_curve.png'))
if test_mean[-1] > test_mean[-2]:
    print("Die Lernkurve zeigt, dass mehr Trainingsdaten wahrscheinlich zu besseren Ergebnissen führen würden.")
else:
    print("Die Lernkurve flacht ab, mehr Daten würden wenig zusätzlichen Nutzen bringen.")

print("\nOptimiertes Modelltraining und -analyse abgeschlossen!")
