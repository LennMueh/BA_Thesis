*Hier stehen die notizen.*

Title: Mutation-Based Accuracy Improvements in Neural Networks using Spectrum-Based Fault Localization

Titel: Mutationsbasierte Genauigkeitsverbesserungen in neuronalen Netzwerken unter Nutzung spektrumbasierter Fehlerlokalisierung.

Heuristikidee: 
- Pick the most suspicious neurons and mutate them.
- Pick the most suspicious neurons by layer and mutate them.

Erste Experimente:
- Löschen bringt nichts, verschlechtert die Genauigkeit.
- Error Rate/Allowance einfügen, als Idee.
- Maybe Choose multiple neurons per epoch. `{1,2,5,10,25}`
- Only 1 epoch is really working.
- Maybe use Loss. (Check if loss is lower than before)