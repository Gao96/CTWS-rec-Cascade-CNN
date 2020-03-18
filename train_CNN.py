import random

import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization,GlobalAveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from keras.callbacks import EarlyStopping
from load_dataset import load_dataset, resize_image, IMAGE_SIZE
from keras.callbacks import TensorBoard, ModelCheckpoint, BaseLogger
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, path_name):
        self.train_images = None
        self.train_labels = None
        self.valid_images = None
        self.valid_labels = None
        self.test_images = None
        self.test_labels = None

        self.path_name = path_name

        self.input_shape = None

    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,img_channels=3, nb_classes=42):
        images, labels = load_dataset(self.path_name)
        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,
                                                                                  random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5,
                                                          random_state=random.randint(0, 100))

        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')

            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
            test_labels = np_utils.to_categorical(test_labels, nb_classes)

            train_images = train_images.astype('float32')
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')

            train_images /= 255
            valid_images /= 255
            test_images /= 255

            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels = test_labels



class Model:
    def __init__(self):
        self.model = None


    def build_model(self, dataset, nb_classes=42):

        self.model = Sequential()

        self.model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=dataset.input_shape,activation="relu")) 
        self.model.add(Convolution2D(64, 3, 3, border_mode='same', activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(3, 3)))
        self.model.add(Dropout(0.2))

        self.model.add(Convolution2D(128, 3, 3,border_mode='same',activation="relu")) 
        self.model.add(Convolution2D(128, 3, 3, border_mode='same',activation="relu"))  
        self.model.add(MaxPooling2D(pool_size=(3, 3))) 
        self.model.add(Dropout(0.2))  

        self.model.add(Flatten()) 
        self.model.add(Dense(4096,activation="relu"))
        self.model.add(Dropout(0.2)) 

        self.model.add(Dense(1024, activation="relu"))  
        self.model.add(Dropout(0.2))

        self.model.add(Dense(nb_classes))  
        self.model.add(Activation('softmax'))  

        self.model.summary()

    def train(self, dataset, batch_size=20, nb_epoch=13, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6,momentum=0.9, nesterov=True)  
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  


        if not data_augmentation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=2)
            history = self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True,callbacks=[early_stopping])

        else:

            datagen = ImageDataGenerator(
                rotation_range=5,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                )

            datagenv = ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                )

            datagen.fit(dataset.train_images)
            datagenv.fit(dataset.valid_images)

 
            history = self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,batch_size=batch_size),
                                     samples_per_epoch=np.ceil(dataset.train_images.shape[0]/batch_size),
                                     nb_epoch=nb_epoch,
                                     validation_data=datagenv.flow(dataset.valid_images, dataset.valid_labels,batch_size=batch_size),
                                     validation_steps=len(datagenv.flow(dataset.valid_images, dataset.valid_labels,batch_size=batch_size))/batch_size
                                    #(dataset.valid_images, dataset.valid_labels),callbacks=[early_stopping]
                                     )

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    MODEL_PATH = ''

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    def predict(self, image):
        
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

        image = image.astype('float32')
        image /= 255


        result1 = self.model.predict_proba(image)
        result = self.model.predict_classes(image)

        return result1[0],result[0]



if __name__ == '__main__':
    dataset = Dataset('D:/pos/')
    dataset.load()

    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model(file_path='C:/Users/64186/PycharmProjects/trafficSignsRecognition/CNN_WarningSignRecognition/warningSigns.model8.h5')

    model = Model()
    model.load_model(file_path='C:/Users/64186/PycharmProjects/trafficSignsRecognition/CNN_WarningSignRecognition/warningSigns.model8.h5')
    model.evaluate(dataset)
    K.clear_session()
