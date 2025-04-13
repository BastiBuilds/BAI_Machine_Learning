import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
data_dir = os.path.join(project_dir, 'data')

# Load data with error handling
file_path = os.path.join(data_dir, 'my_vc_data.csv')
try:
    df = pd.read_csv(file_path)
    print("Dataset shape:", df.shape)
except Exception as e:
    raise Exception(f"Error loading data: {str(e)}")

print("\nMissing values:\n", df.isnull().sum())

# Identify target and features
target = 'Is_Successful'
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset")

# Convert target to numeric and handle any non-numeric values
df[target] = pd.to_numeric(df[target], errors='coerce')

# Identify numeric and categorical columns automatically
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features = [col for col in numeric_features if col != target]
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Convert numeric features to float and handle missing values
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"\nNumeric features ({len(numeric_features)}):", numeric_features)
print(f"Categorical features ({len(categorical_features)}):", categorical_features)

# ------------------------------
# Datenanalyse & Visualisierung
# ------------------------------
print("\nZielverteilung:")
target_distribution = df[target].value_counts(normalize=True)
print(target_distribution)

# Erstelle Visualisierungsordner
viz_dir = os.path.join(project_dir, 'visualizations')
os.makedirs(viz_dir, exist_ok=True)

# 1. Korrelationsanalyse für numerische Features
plt.figure(figsize=(12, 8))
correlation_matrix = df[numeric_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Korrelationsmatrix der numerischen Features')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'correlation_matrix.png'))
plt.close()

# 2. Feature-Verteilungen und Beziehung zum Target
for feature in numeric_features[:6]:  # Erste 6 Features als Beispiel
    plt.figure(figsize=(12, 4))
    
    # Links: Verteilung
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=feature, hue=target, multiple="stack")
    plt.title(f'Verteilung: {feature}')
    
    # Rechts: Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x=target, y=feature)
    plt.title(f'Boxplot: {feature} nach Target')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'distribution_{feature}.png'))
    plt.close()

# 3. Statistische Analyse
print("\nStatistische Analyse der numerischen Features:")
stats_df = pd.DataFrame({
    'Missing': df[numeric_features].isnull().sum(),
    'Mean': df[numeric_features].mean(),
    'Median': df[numeric_features].median(),
    'Std': df[numeric_features].std(),
    'Skew': df[numeric_features].skew()
})
print(stats_df)

# 4. Kategorische Feature-Analyse
if len(categorical_features) > 0:
    print("\nTop 5 häufigste Werte in kategorischen Features:")
    for feature in categorical_features:
        print(f"\n{feature}:")
        print(df[feature].value_counts().head())
        
        # Visualisierung der kategorischen Features
        plt.figure(figsize=(10, 6))
        df_grouped = df.groupby(feature)[target].mean().sort_values(ascending=False)
        df_grouped.plot(kind='bar')
        plt.title(f'Success Rate by {feature}')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'success_rate_{feature}.png'))
        plt.close()

# Definiere X und y
X = df[numeric_features + categorical_features]
y = df[target]

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20, stratify=y
)

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Print missing values before preprocessing
print("\nMissing values before preprocessing:")
print("Training set:", X_train.isnull().sum().sum())
print("Test set:", X_test.isnull().sum().sum())

# Fit preprocessor on training data only
print("\nFitting preprocessor on training data...")
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using training data statistics
print("Transforming test data using training data statistics...")
X_test_processed = preprocessor.transform(X_test)

# Get feature names after preprocessing (fixed version)
numeric_features_final = numeric_features  # Already a list, no .tolist() needed
categorical_features_final = []

try:
    if len(categorical_features) > 0:
        categorical_features_final = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        categorical_features_final = list(categorical_features_final)  # Convert to list if needed
except Exception as e:
    print(f"Warning: Error getting categorical feature names: {e}")
    categorical_features_final = []

all_features = numeric_features_final + categorical_features_final

# Convert to pandas DataFrames
X_train_processed_df = pd.DataFrame(
    X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed,
    columns=all_features
)
X_test_processed_df = pd.DataFrame(
    X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed,
    columns=all_features
)

# Validate no missing values after preprocessing
print("\nMissing values after preprocessing:")
print("Training set:", X_train_processed_df.isnull().sum().sum())
print("Test set:", X_test_processed_df.isnull().sum().sum())

# Add target back to processed data
X_train_processed_df[target] = y_train.values
X_test_processed_df[target] = y_test.values

# Save processed datasets
train_output_path = os.path.join(data_dir, 'processed_train_data.csv')
test_output_path = os.path.join(data_dir, 'processed_test_data.csv')

X_train_processed_df.to_csv(train_output_path, index=False)
X_test_processed_df.to_csv(test_output_path, index=False)

print(f"\nProcessed training data saved to: {train_output_path}")
print(f"Processed test data saved to: {test_output_path}")
print(f"\nTraining set shape: {X_train_processed_df.shape}")
print(f"Test set shape: {X_test_processed_df.shape}")
