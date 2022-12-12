import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
from keras.preprocessing.image import ImageDataGenerator


# Carga de imagenes y muestras
def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    return image[..., ::-1]


new_model = tf.keras.models.load_model('modelo_fast.h5')
dataset_path = "images/"

data_with_aug = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=False,
                                   rescale=1. / 255,
                                   validation_split=0.3)

val = data_with_aug.flow_from_directory(dataset_path,
                                        class_mode="binary",
                                        target_size=(224, 224),
                                        batch_size=98,
                                        subset="validation"
                                        )
predictions = new_model.predict(val)
print(predictions)

val_path = "images/"

plt.figure(figsize=(15, 15))

start_index = 1500

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    preds = np.argmax(predictions[[start_index + i]])
    print(preds)

    gt = val.filenames[start_index + i]
    # [9:13]

    if "angry" in gt:
        gt = 0
    elif 'sad' in gt:
        gt = 1
    elif "relaxed" in gt:
        gt = 2
    else:
        gt = 3

    if preds != gt:
        col = "r"
    else:
        col = "g"

    plt.xlabel('i={}, pred={}, gt={}'.format(start_index + i, preds, gt), color=col)
    plt.imshow(load_img(val_path + val.filenames[start_index + i]))
    plt.tight_layout()

plt.show()
