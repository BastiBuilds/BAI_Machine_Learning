import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# CSV laden und leere Zeilen entfernen
cleaned_csv = pd.read_csv("Datensatz_unternehmen_cleaned.csv")
cleaned_csv = cleaned_csv.dropna()

# Label Encoding für kategoriale Daten
label_encoder = LabelEncoder()
cleaned_csv['state'] = label_encoder.fit_transform(cleaned_csv['state'])
cleaned_csv['BusinessField'] = label_encoder.fit_transform(cleaned_csv['BusinessField'])

# Zielvariable und Inputdaten trennen
set_ohne_zielvariabel = cleaned_csv.drop(columns=["Status"])
zielvariabel = cleaned_csv["Status"]

# Train-Test Split (80/20)
set_ohne_zielvariabel_train, set_ohne_zielvariabel_test, zielvariabel_train, zielvariabel_test = train_test_split(
    set_ohne_zielvariabel, zielvariabel, test_size=0.2, random_state=13)

# Skaling der Daten
scaling = StandardScaler()
set_ohne_zielvariabel_train = scaling.fit_transform(set_ohne_zielvariabel_train)
set_ohne_zielvariabel_test = scaling.transform(set_ohne_zielvariabel_test)

# Label-Encoding für Zielvariable (0 = Failed, 1 = Successful)
zielvariabel_train = label_encoder.fit_transform(zielvariabel_train)
zielvariabel_test = label_encoder.transform(zielvariabel_test)

# Neuronales Netzwerk
model = Sequential()

# Erste Schicht (Eingabeschicht)
model.add(Dense(units=35, activation='relu', kernel_initializer='GlorotNormal', input_dim=set_ohne_zielvariabel_train.shape[1]))

# Weitere Schichten
model.add(Dense(units=100, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(units=150, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(units=200, activation='relu', kernel_initializer='GlorotNormal'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Ausgabeschicht
model.add(Dense(units=1, activation='sigmoid'))

# Kompilieren des Modells
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives()])

# Modell trainieren
history = model.fit(x=set_ohne_zielvariabel_train, y=zielvariabel_train, epochs=700, validation_data=(set_ohne_zielvariabel_test, zielvariabel_test))

# Vorhersagen für Testdaten
y_pred_prob = model.predict(set_ohne_zielvariabel_test)

# Schwellenwert auf 0.35 setzen
y_pred = (y_pred_prob > 0.35).astype(int)  # Erfolgreich, wenn Wahrscheinlichkeit > 0.35

# Confusion Matrix anzeigen
conf_matrix = confusion_matrix(zielvariabel_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Visualisierung der Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Failed', 'Successful'], yticklabels=['Failed', 'Successful'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Neural Network with 0.35 Threshold')
plt.show()

# Klassifikationsbericht (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(zielvariabel_test, y_pred))
