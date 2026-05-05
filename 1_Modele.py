# **************************************************************************
# INF7370 Apprentissage automatique 
# Travail pratique 2 
# ===========================================================================

# #===========================================================================
# Ce modèle est un classifieur (un CNN) entrainé pour classifier des images d'animaux en 6 catégories :
# éléphant, girafe, léopard, rhinocéros, tigre et zèbre.
# Les images sont en couleur (RGB) et redimensionnées à 224×224 pixels.
#
# Données:
# ------------------------------------------------
# entrainement : 6 160 images (80% du dossier entrainement, split automatique)
# validation   : 1 540 images (20% du dossier entrainement, split automatique)
# test         : 1 200 images (200 images par classe)
# ------------------------------------------------

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire

#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Le Type de notre modèle

from keras.models import Model
from keras.models import Sequential

# L'optimisateur Adam utilisé pour ajuster les poids du modèle par descente du gradient

from keras.optimizers import Adam
from keras.regularizers import l2

# Les types des couches utilisées dans notre modèle
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Activation, Dropout, Flatten, Dense, LeakyReLU, GlobalAveragePooling2D

# Des outils pour suivre et gérer l'entrainement de notre modèle
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Sauvegarde du modèle
from keras.models import load_model

# Affichage des graphes 
import matplotlib.pyplot as plt

# Mesure du temps d'exécution
import time


# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
# ==========================================
# ================VARIABLES=================
# ==========================================

# ******************************************************
#                       QUESTION DU TP
# ******************************************************
# 1) Ajuster les variables suivantes selon votre problème:
# - mainDataPath
# - training_batch_size
# - validation_batch_size
# - image_scale
# - image_channels
# - images_color_mode
# - fit_batch_size
# - fit_epochs
# ******************************************************

# Le dossier principal qui contient les données
mainDataPath = "donnees/"
# Le dossier contenant les images d'entrainement
trainPath = mainDataPath + "entrainement"

# Le dossier contenant les images de validation
validationPath = mainDataPath + "validation"

# Le dossier contenant les images de test
testPath = mainDataPath + "test"

# Le nom du fichier du modèle à sauvegarder
modelsPath = "Model.keras"


# Le nombre d'images d'entrainement et de validation
# Il faut en premier lieu identifier les paramètres du CNN qui permettent d’arriver à des bons résultats.
# À cette fin, la démarche générale consiste à utiliser une partie des données d’entrainement et valider
# les résultats avec les données de validation. Les paramètres du réseaux (nombre de couches de convolutions,
# de pooling, nombre de filtres, etc) devrait etre ajustés en conséquence.  Ce processus devrait se répéter
# jusqu’au l’obtention d’une configuration (architecture) satisfaisante. 
# Si on utilise l’ensemble de données d’entrainement en entier, le processus va être long
# car on devrait ajuster les paramètres et reprendre le processus sur tout l’ensemble des données d’entrainement.


training_batch_size = 6160  # 80% de 7700 images (split automatique via validation_split)
validation_batch_size = 1540  # 20% de 7700 images (split automatique via validation_split)

# Configuration des  images 
image_scale = 224 # la taille des images
image_channels = 3  # le nombre de canaux de couleurs (1: pour les images noir et blanc; 3 pour les images en couleurs (rouge vert bleu) )
images_color_mode = "rgb"  # grayscale pour les image noir et blanc; rgb pour les images en couleurs 
image_shape = (image_scale, image_scale, image_channels) # la forme des images d'entrées, ce qui correspond à la couche d'entrée du réseau

# Configuration des paramètres d'entrainement
fit_batch_size = 32 # le nombre d'images entrainées ensemble: un batch
fit_epochs = 60 # Le nombre d'époques 
l2_factor = 1e-4 # coefficient de régularisation L2

# ==========================================
# ==================MODÈLE==================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS DU TP
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Ajuster les deux fonctions:
# 2) feature_extraction
# 3) fully_connected
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Couche d'entrée:
# Prend en entrée des images 224×224×3 (RGB)
input_layer = Input(shape=image_shape)


# Partie feature extraction : 4 blocs de convolution (64→128→256→512 filtres)
# Chaque bloc : Conv2D + ReLU + MaxPooling2D + BatchNormalization
def feature_extraction(input):
  
    # Chaque bloc extrait des caractéristiques de plus en plus complexes :
    # Bloc 1 (64 filtres) : contours et textures de base
    # Bloc 2 (128 filtres) : motifs (rayures, taches)
    # Bloc 3 (256 filtres) : parties d'animaux (oreilles, cornes)
    # Bloc 4 (512 filtres) : formes discriminantes (silhouettes)
    
   
    
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2_factor))(input)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(l2_factor))(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(l2_factor))(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
        
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(l2_factor))(x)
    x = Activation("relu")(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    encoded = BatchNormalization()(encoded)
    return encoded


# Partie entièrement connectée (Fully Connected Layer)
def fully_connected(encoded):
    # GlobalAveragePooling2D : réduit chaque feature map à une seule valeur moyenne
    # Dense(256) + LeakyReLU + BN + Dropout(0.4) : première couche de classification
    # Dense(128) + LeakyReLU + BN + Dropout(0.3) : deuxième couche de classification
    # Dense(6) + Softmax : couche de sortie (une probabilité par classe d'animal)
    x = GlobalAveragePooling2D()(encoded)
    x = Dense(256, kernel_regularizer=l2(l2_factor))(x)
    x = LeakyReLU()(x)    
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
                     
    x = Dense(128, kernel_regularizer=l2(l2_factor))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    # Classification multi-classes (6 classes d'animaux)
    # La dernière couche a 6 neurones (un par classe) avec une fonction d'activation softmax
    # La fonction softmax donne une probabilité pour chaque classe (la somme = 1)
    # La classe prédite est celle avec la probabilité la plus élevée
    x = Dense(6)(x)
    sortie = Activation('softmax')(x)
    return sortie


# Déclaration du modèle:
# La sortie de l'extracteur des features sert comme entrée à la couche complétement connectée
model = Model(input_layer, fully_connected(feature_extraction(input_layer)))

# Affichage des paramètres du modèle
model.summary()

# Compilation du modèle :
# - Fonction de perte : categorical_crossentropy avec label smoothing (0.1) pour améliorer la distinction entre classes similaires
# - Optimisateur : Adam avec learning rate initial de 1e-3
# - Métrique : accuracy
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])

# ==========================================
# ==========CHARGEMENT DES IMAGES===========
# ==========================================

# training_data_generator: charge les données d'entrainement en mémoire
# quand il charge les images, il les ajuste (change la taille, les dimensions, la direction ...) 
# aléatoirement afin de rendre le modèle plus robuste à la position du sujet dans les images
# Note: On peut utiliser cette méthode pour augmenter le nombre d'images d'entrainement (data augmentation)
training_data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    channel_shift_range=25.0,
    fill_mode='nearest',

    # Split validation
    validation_split=0.2
) # on reserve 20% des données d'entrainement pour la validation

# validation_data_generator: charge les données de validation en memoire
validation_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2)

# training_generator: indique la méthode de chargement des données d'entrainement
training_generator = training_data_generator.flow_from_directory(
    trainPath, # Place des images d'entrainement
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),# taille des images
    batch_size=fit_batch_size, # nombre d'images par batch (32)
    class_mode="categorical", # classification multi-classes (6 classes)
    subset="training", # 80% des données pour l'entrainement
    shuffle=True, # on "brasse" (shuffle) les données -> pour prévenir le surapprentissage
    seed=42) # seed pour garantir le même split

# validation_generator: indique la méthode de chargement des données de validation
validation_generator = validation_data_generator.flow_from_directory(
    trainPath, # On utilise le même dossier que l'entrainement (le split est fait automatiquement)
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),  # taille des images
    batch_size=fit_batch_size,  # nombre d'images par batch (32)
    class_mode="categorical",  # classification multi-classes (6 classes)
    subset="validation", # 20% des données pour la validation
    shuffle=False,
    seed=42) # même seed pour garantir le même split

# On imprime l'indice de chaque classe (Keras numerote les classes selon l'ordre des dossiers des classes)
# Dans ce cas => {elephant: 0, girafe: 1, leopard: 2, rhino: 3, tigre: 4, zebre: 5}
print(training_generator.class_indices)
print(validation_generator.class_indices)

# Les générateurs chargent les images par batch directement dans model.fit()
# Cela permet à l'augmentation de données de générer des images différentes à chaque époque
# ==========================================
# ==============ENTRAINEMENT================
# ==========================================

# Savegarder le modèle avec la meilleure validation accuracy ('val_acc') 
# Note: on sauvegarder le modèle seulement quand la précision de la validation s'améliore
modelcheckpoint = ModelCheckpoint(filepath=modelsPath,
                                  monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

# Arrêter l'entrainement si la validation accuracy ne s'améliore plus pendant 10 époques
earlystopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)

# Réduire le learning rate quand la validation accuracy stagne
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# entrainement du modèle
start_time = time.time()
classifier = model.fit(training_generator,
                       epochs=fit_epochs, # nombre d'époques
                       validation_data=validation_generator, # données de validation
                       verbose=1, # mets cette valeur à 0, si vous voulez ne pas afficher les détails d'entrainement
                       callbacks=[modelcheckpoint, earlystopping, reduce_lr]) # les fonctions à appeler à la fin de chaque époque
execution_time = time.time() - start_time

# ==========================================
# ========AFFICHAGE DES RESULTATS===========
# ==========================================

# ***********************************************
#                    QUESTION
# ***********************************************
#
# 4) Afficher le temps d'execution
#
print(f"Temps d'exécution de l'entrainement: {execution_time / 60:.2f} minutes")
# ***********************************************

# Plot accuracy over epochs (precision par époque)
print(classifier.history.keys())
plt.plot(classifier.history['accuracy'])
plt.plot(classifier.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig('accuracy.png', dpi=150, bbox_inches='tight')
plt.show()

# ***********************************************
#                    QUESTION
# ***********************************************
#
# 5) Afficher la courbe d’exactitude par époque (Training vs Validation) ainsi que la courbe de perte (loss)
#
# ***********************************************
plt.plot(classifier.history['loss'])
plt.plot(classifier.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig('loss.png', dpi=150, bbox_inches='tight')
plt.show()