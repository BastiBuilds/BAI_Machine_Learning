import pandas as pd
import numpy as np
from datetime import datetime

# Dateipfade anpassen
input_file_path = 'data/Datensatz_unternehmen.csv'
output_file_path = 'data/Exported_prepared_data_cleaned.csv'

# CSV-Datei korrekt einlesen
print("Lese Daten ein...")
df = pd.read_csv(input_file_path, sep=',')  # Trennzeichen angepasst
print(f"Originale Datenform: {df.shape}")

# Fehlende Werte identifizieren und behandeln
print("\nFehlende Werte pro Spalte vor der Behandlung:")
print(df.isnull().sum())

# Behandlung von fehlenden Werten
for col in df.columns:
    if df[col].dtype == 'object':  # Kategoriale Spalten
        df[col] = df[col].fillna('Unknown')  # Fehlende Werte mit 'Unknown' auffüllen
    elif np.issubdtype(df[col].dtype, np.number):  # Numerische Spalten
        df[col] = df[col].fillna(df[col].median())  # Fehlende Werte mit Median auffüllen
    elif np.issubdtype(df[col].dtype, np.datetime64):  # Datums-Spalten
        df[col] = df[col].fillna(pd.Timestamp('2000-01-01'))  # Fehlende Werte mit einem Standarddatum auffüllen

print("\nFehlende Werte pro Spalte nach der Behandlung:")
print(df.isnull().sum())

# Neue Spalten basierend auf den vorhandenen Daten erstellen
print("\nErstelle neue Spalten...")

# Unternehmensalter berechnen (basierend auf FoundingDate)
if 'FoundingDate' in df.columns:
    df['FoundingDate'] = pd.to_datetime(df['FoundingDate'], errors='coerce', format='%d.%m.%Y')
    df['company_age_years'] = (datetime.now() - df['FoundingDate']).dt.days / 365.25
    df['company_age_years'] = df['company_age_years'].fillna(-1)  # Fehlende Werte mit -1 auffüllen

# Dauer zwischen erstem und letztem Funding
if 'first_funding_date' in df.columns and 'last_funding_date' in df.columns:
    df['first_funding_date'] = pd.to_datetime(df['first_funding_date'], errors='coerce', format='%d.%m.%Y')
    df['last_funding_date'] = pd.to_datetime(df['last_funding_date'], errors='coerce', format='%d.%m.%Y')
    df['funding_duration_years'] = (df['last_funding_date'] - df['first_funding_date']).dt.days / 365.25
    df['funding_duration_years'] = df['funding_duration_years'].fillna(0)  # Fehlende Werte mit 0 auffüllen

# Erfolgsstatus binär kodieren
if 'Status' in df.columns:
    df['is_successful'] = df['Status'].apply(lambda x: 1 if 'Successful' in str(x) else 0)

# Finanzierungsrunden pro Jahr
if 'Fundingrounds' in df.columns and 'funding_duration_years' in df.columns:
    df['funding_rounds_per_year'] = df['Fundingrounds'] / df['funding_duration_years']
    df['funding_rounds_per_year'] = df['funding_rounds_per_year'].replace([np.inf, -np.inf], 0).fillna(0)

# Finanzierung pro Mitarbeiter
if 'Fundingtotal' in df.columns and 'NumContributors' in df.columns:
    df['funding_per_contributor'] = df['Fundingtotal'] / df['NumContributors']
    df['funding_per_contributor'] = df['funding_per_contributor'].replace([np.inf, -np.inf], 0).fillna(0)

# Meilensteine pro Jahr
if 'amount_milestones' in df.columns and 'company_age_years' in df.columns:
    df['milestones_per_year'] = df['amount_milestones'] / df['company_age_years']
    df['milestones_per_year'] = df['milestones_per_year'].replace([np.inf, -np.inf], 0).fillna(0)

# Finale Statistiken anzeigen
print("\nBereinigte Datenform:", df.shape)
print("\nNeue Spalten:")
new_columns = ['company_age_years', 'funding_duration_years', 'is_successful', 'funding_rounds_per_year', 'funding_per_contributor', 'milestones_per_year']
print(new_columns)

# Daten speichern
print(f"\nSpeichere bereinigte Daten unter {output_file_path}...")
df.to_csv(output_file_path, sep=',', index=False)
print("Datenbereinigung abgeschlossen!")