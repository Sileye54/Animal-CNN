# **************************************************************************
# INF7370 Apprentissage automatique 
# Travail pratique 2 
# ===========================================================================

#===========================================================================
# Dans ce script, on évalue le modèle entrainé dans 1_Modele.py
# On charge le modèle en mémoire; on charge les images; et puis on applique le modèle sur les images afin de prédire les classes



# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes
import matplotlib.pyplot as plt

# La librairie numpy 
import numpy as np

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Utilisé pour le calcul des métriques de validation
from sklearn.metrics import confusion_matrix, roc_curve , auc

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model


# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# ==========================================
# ==================MODÈLE==================
# ==========================================

#Chargement du modéle sauvegardé dans la section 1 via 1_Modele.py
model_path = "Model.keras"
Classifier: Model = load_model(model_path)

# ==========================================
# ================VARIABLES=================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 1) A ajuster les variables suivantes selon votre problème:
# - mainDataPath         
# - number_images        
# - number_images_class_x
# - image_scale          
# - images_color_mode    
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# L'emplacement des images de test
mainDataPath = "donnees/"
testPath = mainDataPath + "test"

# Le nombre des images de test à évaluer
number_images = 1200 # 200 images pour chaque classe
number_images_class_elephant = 200
number_images_class_girafe = 200
number_images_class_leopard = 200
number_images_class_rhino = 200
number_images_class_tigre = 200
number_images_class_zebre = 200

# La taille des images à classer (doit correspondre à la taille utilisée pour l'entrainement)
image_scale = 224

# La couleur des images à classer
images_color_mode = "rgb"  # grayscale or rgb

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

# Chargement des images de test
test_data_generator = ImageDataGenerator(rescale=1. / 255)

test_itr = test_data_generator.flow_from_directory(
    testPath,# place des images
    target_size=(image_scale, image_scale), # taille des images
    class_mode="categorical",# Type de classification
    shuffle=False,# pas besoin de les boulverser
    batch_size=1,# on classe les images une à la fois
    color_mode=images_color_mode)# couleur des images

# ==========================================
# ===============ÉVALUATION=================
# ==========================================

# Les classes correctes des images sont récupérées directement depuis le dossier test
class_names = [class_name for class_name, _ in sorted(test_itr.class_indices.items(), key=lambda item: item[1])]
y_true = test_itr.classes.copy()

# evaluation du modele
#test_eval = Classifier.evaluate_generator(test_itr, verbose=1)
test_itr.reset()
test_eval = Classifier.evaluate(test_itr, verbose=1)
# Affichage des valeurs de perte et de precision
print('>Test loss (Erreur):', test_eval[0])
print('>Test précision:', test_eval[1])

# Prédiction des classes des images de test
test_itr.reset()
predicted_probabilities = Classifier.predict(test_itr, verbose=1)

predicted_classes_perc = np.round(predicted_probabilities.copy(), 4)
predicted_classes = np.argmax(predicted_probabilities, axis=1)
# 0 => classe éléphant
# 1 => classe girafe
# 2 => classe léopard
# 3 => classe rhinocéros
# 4 => classe tigre
# 5 => classe zèbre

# Cette list contient les images bien classées
correct = []
for i in range(0, len(predicted_classes) ):
    if predicted_classes[i] == y_true[i]:
        correct.append(i)

# Nombre d'images bien classées
print("> %d  Ètiquettes bien classÈes" % len(correct))

# Cette list contient les images mal classées
incorrect = []
for i in range(0, len(predicted_classes) ):
    if predicted_classes[i] != y_true[i]:
        incorrect.append(i)

# Nombre d'images mal classées
print("> %d Ètiquettes mal classÈes" % len(incorrect))


def show_confusion_matrix(true_labels, predicted_labels, labels):
    matrice = confusion_matrix(true_labels, predicted_labels, labels=np.arange(len(labels)))

    fig, axe = plt.subplots(figsize=(8, 8))
    image = axe.imshow(matrice, cmap="viridis")
    fig.colorbar(image, ax=axe)

    axe.set_title("Confusion Matrix")
    axe.set_xlabel("Predicted label")
    axe.set_ylabel("True label")
    axe.set_xticks(np.arange(len(labels)))
    axe.set_yticks(np.arange(len(labels)))
    axe.set_xticklabels(labels, rotation=45, ha="right")
    axe.set_yticklabels(labels)

    seuil = matrice.max() / 2 if matrice.size > 0 else 0
    for ligne in range(matrice.shape[0]):
        for colonne in range(matrice.shape[1]):
            couleur = "black" if matrice[ligne, colonne] > seuil else "white"
            axe.text(colonne, ligne, matrice[ligne, colonne], ha="center", va="center", color=couleur)

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


def show_wrong_predictions(true_labels, predicted_labels, file_paths, labels):
    nombre_classes = len(labels)
    fig, axes = plt.subplots(nombre_classes, nombre_classes, figsize=(14, 14))

    for true_index in range(nombre_classes):
        for predicted_index in range(nombre_classes):
            axe = axes[true_index, predicted_index]
            indices = np.where((true_labels == true_index) & (predicted_labels == predicted_index))[0]

            if len(indices) > 0:
                image_path = file_paths[indices[0]]
                image = tf.keras.utils.load_img(
                    image_path,
                    color_mode=images_color_mode,
                    target_size=(image_scale, image_scale)
                )
                axe.imshow(image, cmap="gray" if images_color_mode == "grayscale" else None)
            else:
                axe.set_facecolor("#f2f2f2")
                axe.text(0.5, 0.5, "Aucune\nimage", ha="center", va="center", fontsize=8)

            axe.set_xticks([])
            axe.set_yticks([])

            if true_index == 0:
                axe.set_title(labels[predicted_index], fontsize=10)

            if predicted_index == 0:
                axe.set_ylabel(labels[true_index], fontsize=10)

    fig.suptitle("Une image par combinaison vrai/predit", fontsize=14)
    plt.tight_layout()
    plt.savefig('wrong_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 1) Afficher la matrice de confusion
# 2) Extraire une image mal-classée pour chaque combinaison d'espèces - Voir l'exemple dans l'énoncé.
# ***********************************************

show_confusion_matrix(
    y_true,
    predicted_classes,
    class_names
)

show_wrong_predictions(
    y_true,
    predicted_classes,
    np.array(test_itr.filepaths),
    class_names
)
