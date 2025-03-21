import pandas as pd
import numpy as np
from datetime import datetime

# Dateipfad anpassen
input_file_path = 'data/Exported_prepared_data.csv'
output_file_path = 'data/Exported_prepared_data_cleaned.csv'

# Daten einlesen (Semikolon als Trennzeichen)
print("Lese Daten ein...")
df = pd.read_csv(input_file_path, sep=';')

print(f"Originale Datenform: {df.shape}")

# Fehlende Werte identifizieren
missing_values = df.isnull().sum()
print("\nFehlende Werte pro Spalte:")
print(missing_values[missing_values > 0])

# Hilfsfunktion zum Umwandeln des Datums im Format DD.MM.YYYY in Timestamp
def convert_date(date_str):
    if isinstance(date_str, str):
        try:
            return datetime.strptime(date_str, '%d.%m.%Y')
        except:
            return pd.NaT
    return pd.NaT

# Datum-Spalten umwandeln
date_columns = ['first_milestone_date', 'last_funding_date', 'first_funding_date', 
                'last_milestone_date', 'FoundingDate', 'closedDate', 'acquiredDate']

for col in date_columns:
    if col in df.columns:
        df[col] = df[col].apply(convert_date)
        
        # Neue numerische Features aus den Datumsfeldern erstellen
        # Umwandlung in Tage seit Anfang 2000
        reference_date = datetime(2000, 1, 1)
        new_col_name = f"{col}_days_since_2000"
        df[new_col_name] = (df[col] - reference_date).dt.days
        
        # Fehlende Werte mit -1 kennzeichnen
        df[new_col_name] = df[new_col_name].fillna(-1)

# Feature Engineering: Neue Zeitspanenfeatures erstellen
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

# Fehlende Werte im Unternehmensdauer-Feature füllen (sollte eigentlich nicht vorkommen)
df['company_age_at_outcome_days'] = df['company_age_at_outcome_days'].fillna(-1)

# Finanzierungseffizienz: Geld pro Tag Unternehmensleben
df['funding_per_day'] = df.apply(
    lambda row: row['Fundingtotal'] / row['company_age_at_outcome_days'] 
    if row['company_age_at_outcome_days'] > 0 else 0, 
    axis=1
)

# Verhältnis von Finanzierung zu Mitarbeitern
df['funding_per_contributor'] = df.apply(
    lambda row: row['Fundingtotal'] / row['NumContributors']
    if row['NumContributors'] > 0 else 0,
    axis=1
)

# Finanzierungsrunden pro Jahr
df['founding_to_outcome_years'] = df['company_age_at_outcome_days'] / 365.25
df['funding_rounds_per_year'] = df.apply(
    lambda row: row['Fundingrounds'] / row['founding_to_outcome_years']
    if row['founding_to_outcome_years'] > 0 else 0,
    axis=1
)

# Meilensteine pro Jahr (falls vorhanden)
df['milestones_per_year'] = df.apply(
    lambda row: row['amount_milestones'] / row['founding_to_outcome_years']
    if pd.notna(row['amount_milestones']) and row['founding_to_outcome_years'] > 0 else 0,
    axis=1
)

# Meilenstein-bezogene Features behandeln
milestone_features = ['average_milestone_duration', 'average_milestone_duration_days', 
                     'milestone_days_duration', 'years_from_first_to_last_milestone',
                     'amount_milestones']

for col in milestone_features:
    if col in df.columns:
        # Fehlende Werte mit 0 füllen, da fehlende Meilensteine eine Information sind
        df[col] = df[col].fillna(0)

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

# Kategoriale Features für Geschäftsfelder mit weniger als 5 Startups zusammenfassen
business_field_counts = df['BusinessField'].value_counts()
rare_fields = business_field_counts[business_field_counts < 5].index
df['BusinessField_grouped'] = df['BusinessField'].apply(lambda x: 'OTHER' if x in rare_fields else x)

# Überprüfen auf Ausreißer in numerischen Daten und Winsorisieren
numeric_cols = ['Fundingtotal', 'funding_per_day', 'funding_per_contributor', 
                'NumContributors', 'NumPartners']

for col in numeric_cols:
    # 1% und 99% Perzentile berechnen
    lower_bound = df[col].quantile(0.01)
    upper_bound = df[col].quantile(0.99)
    
    # Ausreißer begrenzen (Winsorizing)
    df[col] = df[col].clip(lower_bound, upper_bound)

# Z-Scores für wichtige numerische Features berechnen und hinzufügen
for col in ['Fundingtotal', 'Fundingrounds', 'NumContributors', 'NumPartners', 
            'funding_per_day', 'milestones_per_year']:
    mean_val = df[col].mean()
    std_val = df[col].std()
    if std_val > 0:  # Vermeiden einer Division durch Null
        df[f'{col}_zscore'] = (df[col] - mean_val) / std_val

# Log-Transformation für schiefe Verteilungen
for col in ['Fundingtotal', 'NumContributors']:
    # Kleinen Wert hinzufügen, um log(0) zu vermeiden
    df[f'{col}_log'] = np.log1p(df[col])

# Spalten entfernen, die für das Modell nicht nützlich sind
columns_to_drop = ['Is Duplicate Row?', 'LocalCode:42']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Finale Statistiken anzeigen
print("\nBereinigte Datenform:", df.shape)
print("\nNeue Features:")
new_columns = set(df.columns) - set(pd.read_csv(input_file_path, sep=';').columns)
print(sorted(list(new_columns)))

# Daten speichern
print(f"\nSpeichere bereinigte Daten unter {output_file_path}...")
df.to_csv(output_file_path, sep=';', index=False)

print("Datenbereinigung abgeschlossen!")
