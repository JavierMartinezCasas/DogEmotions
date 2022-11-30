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

# Path de las imagenes
angry = 'images_cort/angry/'
#sad = 'images/sad/'
relaxed = 'images_cort/relaxed/'
happy = 'images_cort/happy/'

angry_path = os.listdir(angry)
#sad_path = os.listdir(sad)
relaxed_path = os.listdir(relaxed)
happy_path = os.listdir(happy)


# Carga de imagenes y muestras
def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    return image[..., ::-1]


fig = plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(angry + angry_path[i]), cmap='gray')
    plt.suptitle("Angry Dogs", fontsize=20)
    plt.axis('off')

plt.show()
"""
fig = plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(sad + sad_path[i]), cmap='gray')
    plt.suptitle("Sad Dogs", fontsize=20)
    plt.title(sad_path[i][:4])
    plt.axis('off')

plt.show()
"""
fig = plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(relaxed + relaxed_path[i]), cmap='gray')
    plt.suptitle("Relaxed Dogs", fontsize=20)
    plt.title(relaxed_path[i][:4])
    plt.axis('off')

plt.show()

fig = plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(happy + happy_path[i]), cmap='gray')
    plt.suptitle("Happy Dogs", fontsize=20)
    plt.title(happy_path[i][:4])
    plt.axis('off')

plt.show()

# Datasets
dataset_path = "images_cort"

data_with_aug = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=False,
                                   rescale=1. / 255,
                                   validation_split=0.3)

train = data_with_aug.flow_from_directory(dataset_path,
                                          class_mode="binary",
                                          target_size=(96, 96),
                                          batch_size=32,
                                          subset="training")

val = data_with_aug.flow_from_directory(dataset_path,
                                        class_mode="binary",
                                        target_size=(96, 96),
                                        batch_size=32,
                                        subset="validation"
                                        )

mnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))

tf.keras.backend.clear_session()

model = Sequential([mnet,
                    GlobalAveragePooling2D(),
                    Dense(512, activation="relu"),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(128, activation="relu"),
                    Dropout(0.1),
                    # Dense(32, activation = "relu"),
                    # Dropout(0.3),
                    Dense(4, activation="sigmoid")])

model.layers[0].trainable = False

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics="accuracy")

model.summary()
Model: "sequential"


def scheduler(epoch):
    if epoch <= 2:
        return 0.001
    elif 2 < epoch <= 15:
        return 0.0001
    else:
        return 0.00001


lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)

hist = model.fit_generator(train,
                           epochs=20,
                           callbacks=[lr_callbacks],
                           validation_data=val)

epochs = 20
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'])
# print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
# print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()

train = data_with_aug.flow_from_directory(dataset_path,
                                          class_mode="binary",
                                          target_size=(224, 224),
                                          batch_size=98,
                                          subset="training")

val = data_with_aug.flow_from_directory(dataset_path,
                                        class_mode="binary",
                                        target_size=(224, 224),
                                        batch_size=98,
                                        subset="validation"
                                        )

vgg16_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

vgg16_model.output[-1]

# model = Sequential()
# for layer in vgg16_model.layers[:-1]:
#     model.add(layer)

# for layer in model.layers:
#     layer.trainable = False

# model.add(Dense(2, activation='softmax'))


model = Sequential([vgg16_model,
                    Flatten(),
                    #                     GlobalAveragePooling2D(),
                    #                     Dense(512, activation = "relu"),
                    #                     BatchNormalization(),
                    #                     Dropout(0.3),
                    #                     Dense(128, activation = "relu"),
                    #                     Dropout(0.1),
                    #                     # Dense(32, activation = "relu"),
                    #                     # Dropout(0.3),
                    Dense(4, activation="sigmoid")])

model.layers[0].trainable = False

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics="accuracy")

model.summary()

hist = model.fit_generator(train,
                           epochs=20,
                           callbacks=[lr_callbacks],
                           validation_data=val)


model.save('modelo_fast.h5')
epochs = 20
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'])
# print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
# print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

# Creating an array of predicted test images
predictions = model.predict(val)

val_path = "images/"

plt.figure(figsize=(15, 15))

start_index = 250

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    preds = np.argmax(predictions[[start_index + i]])
    print(preds)
    gt = val.filenames[start_index + i]

    if "angry" in gt:
        gt = 0
    elif 'happy' in gt:
        gt = 1
    elif 'relaxed' in gt:
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
