import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from IPython.display import display
from PIL import Image

TRAIN_DIR = "D:\\Projects\\Project X\\dataset\\training_set"
TEST_DIR = "D:\\Projects\\Project X\\dataset\\testing_set"

CATEGORIES = ["akiec", "bcc", "bkl", "mel", "nv", "vasc"]
IMG_SIZE = 50
nb_classes = 6
batch_size = 128

training_data = []

#loading data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(64,64),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(64,64),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical'
)


datagen_train = ImageDataGenerator()
datagen_validation = ImageDataGenerator()

train_generator = datagen_train.flow_from_directory(TRAIN_DIR,
                                                    target_size=(64,64),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = datagen_validation.flow_from_directory(TEST_DIR,
                                                    target_size=(64,64),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

#initializes the training data
def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(TRAIN_DIR,category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


#trains the model
def train_model():
    model = Sequential()

    # 1 - Convolution
    model.add(Conv2D(64,(3,3), padding='same', input_shape=(64, 64,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd Convolution layer
    model.add(Conv2D(128,(5,5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # 3rd Convolution layer
    model.add(Conv2D(512,(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flattening
    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Fully connected layer 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes, activation='softmax'))

    opt = Adam(lr=0.0001)

    model.compile(loss='categorical_crossentropy',
                optimizer= opt,
                metrics=['accuracy'])


    #model.fit(training_data, training_data, batch_size=32, epochs=3, validation_split=0.3)

    checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.n//train_generator.batch_size,
        epochs=50,
        validation_data = validation_generator,
        validation_steps = validation_generator.n//validation_generator.batch_size,
        callbacks=callbacks_list
    )

    # serialize model structure to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save('trained_model.h5')

#saves the model on its first train
def save_model():
    pass

if __name__ == "__main__":

    #create_training_data()
    train_model()