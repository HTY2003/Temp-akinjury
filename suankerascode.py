import os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

nb_train_samples = 400
nb_validation_samples = 100
epochs = 10
batch_size = 16

paths = ['cleanedhealine/firstdegburn/', \
            'cleanedhealine/minorcut/', \
            'cleanedhealine/contusion/', \
            'cleanedhealine/snakebite/', \
            'cleanedhealine/nosebleed/']
files = []
for path in paths:
    classlist = [path+f for f in listdir(path) if isfile(join(path, f))]
    classlist = [f for f in classlist if f.endswith('.jpg') or f.endswith('.png')or f.endswith('.jpeg')]
    files.append(classlist)

data = []
for clasz in files:
    classdata = []
    for file in clasz:
        array1 = np.array(Image.open(file))
        array2 = np.flipud(array1)
        array3 = np.fliplr(array1)
        array4 = np.rot90(array1, 2)
        classdata.append(array1)
        classdata.append(array2)
        classdata.append(array3)
        classdata.append(array4)
        data.append(classdata)
X = []
Y = []
for i in range(5):
    print(i)
    for arr in data[i]:
        X.append(arr/255)
        Y.append(i)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

img_width, img_height = 224, 224
nb_train_samples = len(Y_train)
nb_validation_samples = len(Y_test)
epochs = 10
batch_size = 16
input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
