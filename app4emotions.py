import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter import filedialog
import matplotlib

matplotlib.use('TkAgg')
new_model = tf.keras.models.load_model('modelo2.h5')


archivo = filedialog.askopenfilename(title="abrir", initialdir="D:/Descargas")

image = tf.keras.preprocessing.image.load_img(archivo, target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.

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
