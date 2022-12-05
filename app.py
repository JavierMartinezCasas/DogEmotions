import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dropout, Dense, BatchNormalization, Flatten
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

new_model = tf.keras.models.load_model('modelo_fast.h5')

import tkinter as tk
import tkinter.filedialog
from tkinter import filedialog
from tkinter import *

archivo = filedialog.askopenfilename(title="abrir", initialdir="D:/Descargas", filetype=(("Todos los archivos", "*.*"),
                                                                                         ("Archivos de Texto", "*.txt"),
                                                                                         ("Archivos Word", "*.docx"),
                                                                                         ("Archivos json", "*.json")))

# Button(text="Selecciona imagen", background="pale green", command=sel_img).place(x=10, y=10)
# archivo = filedialog.askdirectory(title="abrir", initialdir="D:/Descargas")
image = tf.keras.preprocessing.image.load_img(archivo, target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.

new_model = tf.keras.models.load_model('modelo_fast.h5')  # Es el modelo corto actualizar abajo
predictions = new_model.predict(input_arr)
preds = np.argmax(predictions)
if preds == 0:
    predi = "Enfadado"
    print("Enfadado")
elif preds == 1:
    predi = "Triste"
    print("Triste")
elif preds == 2:
    predi = "Relajado"
    print("Relajado")
else:
    predi = "Contento"
    print("Contento")

plt.xlabel(predi, color="g")
plt.imshow(image)
plt.tight_layout()

plt.show()
