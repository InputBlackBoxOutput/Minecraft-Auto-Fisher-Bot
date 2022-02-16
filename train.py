import os, re, random, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models, optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split

WIDTH = 160
HEIGHT = 90

def load_dataset_image_paths(dataset_path="./dataset"):
	_float= glob.glob(dataset_path + "/float/*")
	random.shuffle(_float)
	_sunk = glob.glob(dataset_path + "/sunk/*")
	random.shuffle(_sunk)

	return _float, _sunk

def preprocess(images, target):
    x = []
    y = []
    
    for image in images:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        
        x.append(img)
    
    return x, [target] * len(images)

def get_model():
	model = models.Sequential()

	model.add(layers.Conv2D(32, (3, 3), input_shape=(HEIGHT, WIDTH, 3)))
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))

	model.add(layers.Conv2D(16, (3, 3)))
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))

	model.add(layers.Conv2D(8, (3, 3)))
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))

	model.add(layers.Conv2D(4, (3, 3)))
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))

	model.add(layers.Flatten())
	model.add(layers.Dense(64))
	model.add(layers.Dense(16))
	model.add(layers.Dense(4))
	model.add(layers.Dense(1))
	model.add(layers.Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	model.summary()
	return model

def get_train_generator(X_train, Y_train):
	train_datagen = ImageDataGenerator(
		rescale=1. / 255, 
		shear_range=0.2, 
		zoom_range=0.1, 
		horizontal_flip=True)
	
	train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size=batch_size)

	return train_generator

def get_val_generator(X_val, Y_val):
	val_datagen = ImageDataGenerator(
	    rescale=1. / 255,
	    shear_range=0.2,
	    zoom_range=0.1,
	    horizontal_flip=True)

	validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size=batch_size)

	return validation_generator



if __name__ == "__main__":
	_float, _sunk = load_dataset_image_paths()
	print(f"Number of float samples: {len(_float)}")
	print(f"Number of sunk samples: {len(_sunk)}")

	X, Y = preprocess(_float, target=0)
	_X, _Y = preprocess(_sunk, target=1)
	X += _X
	Y += _Y

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
	X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)

	nb_train_samples = len(X_train)
	nb_validation_samples = len(X_val)

	epochs = 20
	batch_size = 16

	print("Model:")
	model = get_model()
	train_generator = get_train_generator(X_train, Y_train)
	validation_generator = get_val_generator(X_val, Y_val)

	print("Train:")
	history = model.fit(
    	train_generator, 
	    steps_per_epoch=nb_train_samples // batch_size,
	    epochs=epochs,
	    validation_data=validation_generator,
	    validation_steps=nb_validation_samples // batch_size
	)

	print("Test:")
	model.evaluate(np.array(X_test)/255, np.array(Y_test))
	model.save('model.h5')