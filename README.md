# BAI_Machine_Learning
Szenario
MyVC ist eine Venture Capital-Firma, die das Geld ihrer Kunden als Risikokapital in Start-ups in den
USA investiert. Die Kernaufgabe von MyVC ist es abzuschätzen, welche Start-ups es schaffen werden,
am Ende für eine hohe Summe gekauft zu werden. Dazu hat MyVC Daten aus der Vergangenheit
gesammelt und möchte nun den Erfolg zukünftiger Investments vorhersagen. Die Daten zu den
Startups wurden zunächst am Ende des Jahres 2014 gesammelt. Dann wurde ein Zeitraum von 5
Jahren abgewartet und festgestellt, was aus den Startups jeweils geworden war, d.h. ob sie
akquiriert worden waren oder schliessen mussten.
Aufgabe
• Formalisiert die Aufgabe als Klassifikations- oder Regressionsaufgabe, d.h. legt fest, wie die
Instanzen, die Attribute und die Zielvariable bzw. das Klassenattribut definiert sind. Falls ihr
neue Attribute definiert, beschreibt bitte, wie diese berechnet werden. Falls ihr Attribute
aus den gegebenen Daten nicht verwendet, begründet bitte, warum nicht. Beschreibt bitte
auch allfällige Typkonversionen, die ihr für bestehende Attributen vorschlagt!
• Implementiert die Aufbereitung der Daten, z.B. in Tableau Prep oder in Orange oder Python.
Die Aufbereitung der Daten kann die Verknüpfung verschiedener Daten beinhalten, die
Berechnung neuer Attribute (inklusive Zielvariablen) usw.
• Wählt einen geeigneten Klassifikations- bzw. Regressionsalgorithmus aus und wendet ihn
auf die Daten an. Natürlich könnt ihr auch mit mehreren Algorithmen experimentieren.
Entscheidet euch am Ende für einen Algorithmus und begründet eure Wahl!
• Falls ihr bestimmte Techniken wie Regularisierung, Over-/Undersampling oder
Dimensionsreduktion anwendet, begründet dies bitte und beschreibt, wie ihr es umsetzt.
• Beschreibt so gut wie möglich die Muster, die euer Algorithmus aus den Daten gelernt hat.
Ordnet diese Muster nach ihrer Wichtigkeit für MyVC.
• Erarbeitet eine geeignete Evaluationsstrategie, d.h. wählt eine Prozedur (z.B. Holdout oder
Kreuzvalidierung), sowie ein zu optimierendes Mass aus (z.B. Accuracy, F-Measure, AUC oder
Kostenmatrix). Begründet eure Wahl! Beschreibt dann, wie gut euer Algorithmus bezüglich
des gewählten Evaluationsmasses abschneidet und was dies für die Anwendung bei MyVC
bedeutet. Welchen ökonomischen Nutzen hat das gelernte Modell?
Lieferobjekte
Abzugeben sind die folgenden Lieferobjekte:
- Eine Präsentation, welche die oben beschriebenen Punkte abdeckt, d.h. angefangen mit der
Formalisierung der Aufgabe und der Definition und Begründung aller Attribute, über die
Beschreibung und Begründung der Datenaufbereitung und der Wahl der auf die Daten
angewendeten Algorithmen bis hin zur Evaluationsstrategie und Diskussion der
Hans Friedrich Witschel, Andreas Martin
Maschinelles Lernen
Evaluationsergebnisse. Die Präsentation soll so präzise und kurz wie möglich die genannten
Punkte abdecken!
- Den Workflow für die Datenaufbereitung (z.B. Tableau Prep oder Orange-Workflow oder
Python-Code)
- Den Orange-Workflow für die Anwendung und Evaluation der gewählten Klassifikations-
bzw. Regressionsalgorithmen. Falls ihr für die Datenaufbereitung Orange verwendet, könnt
ihr auch nur einen Workflow abgeben.
- Die Input-Datei im csv-Format, welche eurer Formalisierung der Aufgabe entspricht und als
Input für euren Orange-Workflow dient.