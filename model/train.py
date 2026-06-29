import random, glob
import cv2
import numpy as np

from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models
from sklearn.model_selection import train_test_split

WIDTH  = 320
HEIGHT = 320

EPOCH = 20
BATCH = 16

def preprocess(images):
    x = []
    
    for image in images:
        img = cv2.imread(image)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        
        x.append(img)
    
    return np.array(x, dtype=np.float32)

def model():
    model = models.Sequential()

    model.add(layers.InputLayer(input_shape=(HEIGHT, WIDTH, 3)))
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation("relu"))
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
    model.add(layers.Dense(128))
    model.add(layers.Dense(64))
    model.add(layers.Dense(16))
    model.add(layers.Dense(4))
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    model.summary()
    return model

if __name__ == "__main__":
    _float = glob.glob("../dataset/float/*")
    random.shuffle(_float)
    _float = preprocess(_float)
    print(f"Number of float samples: {len(_float)}")

    _sunk  = glob.glob("../dataset/sunk/*")
    random.shuffle(_sunk)
    _sunk = preprocess(_sunk)
    print(f"Number of sunk samples: {len(_sunk)}")

    X = np.vstack([_float, _sunk])
    Y = np.array(([0] * len(_float)) + ([1] * len(_sunk)))

    X_train, X_test, Y_train, Y_test = train_test_split(
		X, Y, 
		test_size=0.2, 
		random_state=1
	)

    X_train, X_validation, Y_train, Y_validation = train_test_split(
		X_train, Y_train, 
		test_size=0.2, 
		random_state=1
	)

    print("Model:")
    m = model()

    train_generator = ImageDataGenerator(
		rescale=1.0 / 255, 
		zoom_range=0.25
	).flow(
		X_train, 
		Y_train, 
		batch_size=BATCH
	)

    validation_generator = ImageDataGenerator(
		rescale=1.0 / 255, 
		zoom_range=0.25
	).flow(
		X_validation, 
		Y_validation, 
		batch_size=BATCH
	)

    print("Train:")
    history = m.fit(
        train_generator,
        steps_per_epoch=len(X_train) // BATCH,
        epochs=EPOCH,
        validation_data=validation_generator,
        validation_steps=len(X_validation) // BATCH
    )

    print("Test:")
    m.evaluate(
		np.array(X_test)/255.0, 
		np.array(Y_test)
	)
    m.save('./model.h5')