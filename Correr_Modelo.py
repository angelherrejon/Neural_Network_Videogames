from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Cargar el modelo entrenado
model = load_model('modelo_videogames.h5')

# Preparar la imagen de entrada
ruta_imagen = '01.jpg'
img = image.load_img(ruta_imagen, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Hacer la predicción
prediccion = model.predict(img_array)
indice_predicho = np.argmax(prediccion)

# Clases en orden alfabético
clases = [
    'A_Plague_Tale',
    'Call_of_Duty',
    'League_of_Legends',
    'Minecraft',
    'Mortal_Kombat',
    'Resident_Evil',
    'Shadow_of_The_Tomb_Raider',
    'Super_Mario_Galaxy',
    'The_Legend_of_Zelda',
    'Valorant'
]

# Mostrar resultado
nombre_clase = clases[indice_predicho]
confianza = prediccion[0][indice_predicho] * 100

# Mostrar la imagen con matplotlib
plt.imshow(img)
plt.title(f"Predicción: {nombre_clase} ({confianza:.2f}%)", fontsize=14)
plt.axis('off')
plt.show()

