# Animal-CNN

Classification d'images d'animaux par réseau de neurones convolutif (CNN) — 6 espèces : éléphant, girafe, léopard, rhinocéros, tigre et zèbre.

> **Exactitude sur les données de test : 92.62%**

---

## Description

Ce projet implémente un CNN from scratch avec Keras/TensorFlow pour classifier des images d'animaux sauvages en 6 catégories. Le modèle est entraîné sur ~7 700 images RGB redimensionnées à 224×224 pixels, avec augmentation de données pour améliorer la généralisation.

---

## Architecture du modèle

Le CNN est composé de **4 blocs de convolution** suivis de **couches entièrement connectées** :

### Extracteur de caractéristiques

| Bloc | Filtres | Noyau | Activation | Pooling | BatchNorm |
|:----:|:-------:|:-----:|:----------:|:-------:|:---------:|
| 1    | 64      | 3×3   | ReLU       | MaxPool 2×2 | ✅ |
| 2    | 128     | 3×3   | ReLU       | MaxPool 2×2 | ✅ |
| 3    | 256     | 3×3   | ReLU       | MaxPool 2×2 | ✅ |
| 4    | 512     | 3×3   | ReLU       | MaxPool 2×2 | ✅ |

### Couches entièrement connectées

| Couche | Neurones | Activation | Dropout |
|:------:|:--------:|:----------:|:-------:|
| GlobalAveragePooling2D | — | — | — |
| Dense | 256 | LeakyReLU | 0.4 |
| Dense | 128 | LeakyReLU | 0.3 |
| Dense (sortie) | 6 | Softmax | — |

---

## Données

| Catégorie    | Entraînement (80%) | Validation (20%) | Test |
|:------------:|:------------------:|:----------------:|:----:|
| Éléphant     | ~1 027 | ~257 | 200 |
| Girafe       | ~1 027 | ~257 | 200 |
| Léopard      | ~1 027 | ~257 | 200 |
| Rhinocéros   | ~1 027 | ~257 | 200 |
| Tigre        | ~1 027 | ~257 | 200 |
| Zèbre        | ~1 027 | ~257 | 200 |
| **Total**    | **~6 160** | **~1 540** | **1 200** |

Les images sont placées dans `donnees/entrainement/` et `donnees/test/`, organisées par sous-dossiers (une classe par dossier).

---

## Hyperparamètres

| Paramètre | Valeur |
|:-----------|:-------|
| Optimiseur | Adam (lr = 1×10⁻³) |
| Fonction de perte | Categorical Crossentropy (label smoothing = 0.1) |
| Batch size | 32 |
| Époques max | 60 |
| Early Stopping | patience = 10 sur `val_accuracy` |
| ReduceLROnPlateau | facteur ×0.5, patience = 5 |
| Régularisation L2 | 1×10⁻⁴ |

### Augmentation de données

Rotation (±15°), décalage horizontal/vertical (±10%), cisaillement, zoom (±15%), retournement horizontal, variation de luminosité, et décalage de canaux couleur (±25.0).

---

## Installation et utilisation

### Prérequis

- Python 3.8+
- GPU recommandé (NVIDIA avec CUDA)

### Installation

```bash
git clone https://github.com/Sileye54/Animal-CNN.git
cd Animal-CNN
python -m venv .venv
source .venv/bin/activate
pip install -r requierements.txt
```

### Entraînement

```bash
python 1_Modele.py
```

Le meilleur modèle est sauvegardé automatiquement dans `Model.keras`.

### Évaluation

```bash
python 2_Evaluation.py
```

Génère la matrice de confusion et les exemples de prédictions incorrectes.

---

## Structure du projet

```
Animal-CNN/
├── 1_Modele.py              # Entraînement du CNN
├── 2_Evaluation.py          # Évaluation sur les données de test
├── Model.keras              # Modèle sauvegardé (meilleur val_accuracy)
├── requierements.txt        # Dépendances Python
├── rapport.md               # Rapport détaillé du projet
├── donnees/                 # Données (non incluses dans le repo)
│   ├── entrainement/        # Images d'entraînement (par classe)
│   └── test/                # Images de test (par classe)
├── accuracy.png             # Courbe d'exactitude
├── loss.png                 # Courbe de perte
├── confusion_matrix.png     # Matrice de confusion
├── wrong_predictions.png    # Exemples d'erreurs de classification
└── my_arch-1.png            # Schéma de l'architecture
```


---
