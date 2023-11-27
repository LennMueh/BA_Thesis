# Jour Fixe Daniel/Lennart

## Questions Guide
1. Was waren die Ziele der letzten Woche?
2. Wurden die Ziele erreicht?
3. Gab es Beobachtungen von Interesse?
4. Gab es Probleme?
5. Mussten Schritte angepasst werden, um die Ziele zu erreichen?
6. Gibt es ungelöste Probleme?
7. Was sind die nächsten Ziele?
8. Was sind die nächsten Schritte für diese Ziele?

## Notizen

### 17.11.2023

#### Vorbereitung
1. Funktionen und anderes
   - Analyse Funktionen
   - Vorbereitende Funktionen der Analyse
   - "Run" Funktion
   - Funktion zum Löschen von Neuronen
   - 1 Epochen Modelle
   - Strukturabriss
2. Teilweise, es fehlen:
    - Funktion zum Löschen von Neuronen
    - "Run" Funktion
3. Nein, aber: Funktion zum Zufälligen wählen von Neuronen vorgezogen
4. Ja, folgendes:
   - Strukturabriss, Main Chapter was gehört rein?
   - Gedanke, wie Experimente gestalten?
        - Wie viele Experimente?
        - Wie aufbereiten, Loss und Accuracy?
   - Gedanke, wie Ergebnisse darstellen?
     - Verschidene Größen von Neuronalen Netzen
     - Vielleicht zu viele verschiedene Größen?
5. Nein
6. Nein
7. Funktion zum Modifizieren von Neuronen
   - Löschen von Neuronen
   - Modifizieren von Weight
   - Modifizieren von Bias
8. Implementierung der Funktionen

#### Notizen
In 2 Wochen MainChapter Struktur vorstellen (1.12.2023) \
Nächste Woche alten Projektplan finden und aktualisieren (24.11.2023)

### 24.11.2023

#### Vorbereitung
1. Funktionen and run first experiments
   - Funktion zum Löschen und Modifizieren von Neuronen
   - Erste Durchläufe
2. Ja
3. Ja
    - Algorithmus bricht schnell ab
    - 6 Hid layers bringt kaum eine Verbesserung
    - Veränderungen sehen sehr random aus
    - Veränderungen sind sehr klein
4. Nein, keine Anhaltenden Probleme
5. Nein, noch nicht
6. Nein
7. Weitere Funktionalitäten
    - Mehrere Neuronen auf einmal modifizieren und trainieren
    - Error Rate/Allowance einfügen
    - Loss als Kriterium
    - 6 Layer Modelle nicht mehr benutzen
8. Implementierung in der Funtion zum Testen

#### Notizen
Ideen für Experimente:
1. [ ] Versuche Convolutions zu prüfen
2. [ ] Faktor multiplizieren Weight und Bias `{-1,-0.5,-0.25,0.25,0.5,1}`
3. [ ] Kleinere Datensätze fürs Training
4. [x] Remove the gelu models
5. [ ] add models with less neurons per layer, with more layers
6. [ ] Dann, verdächtigsten Layer herausfinden und diesen Löschen
7. [ ] Dann, verdächtigsten Layer herausfinden und diesen einen neuen Layer hinterstellen/vorstellen (dense, convolution)
8. [ ] Normal and uniform distribution in random neuron bias and weight modification