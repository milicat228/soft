#za neuronsku mrežu
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.models import model_from_json

#za proveru postoji li fajl
import os.path

#za pretprocesiranje slika
import cv2
import numpy as np

class NeuralNetwork:
    model = None

    def __init__(self):
        """Učitava model sa diska, ako postoji, a ako ne postoji kreira novi."""
        if os.path.isfile('model/model.json') and os.path.isfile('model/model.h5'):
            json_file = open('model/model.json', 'r')
            self.model = json_file.read()
            json_file.close()
            self.model = model_from_json(self.model)
            self.model.load_weights("model/model.h5")
        else :
            data = self.get_data_ready()
            self.train(data['x_train'],data['x_test'],data['y_train'],data['y_test'])
            self.save_model()

    def get_data_ready(self):
        """Metoda učitava podatke i pretprocesira ih."""
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        #oseci crne ivice test i train slikama, a zatim ih vrati na 28x28
        x_temp_train = []
        for image in X_train:
            x_temp_train.append(self.process_image(image))            
        x_train = np.array(x_temp_train, ndmin=1)

        x_temp_test = []
        for image in X_test:
            x_temp_test.append(self.process_image(image))            
        x_test = np.array(x_temp_test, ndmin=1)

        #pretvori slike iz matrica u nizove
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        #normalizacija vrednosti na ospeg 0-1
        x_train /= 255
        x_test /= 255 
        #obrada labela (npr 5 se pretvara u 0000010000)  
        classes = 10
        y_train = np_utils.to_categorical(y_train, classes)
        y_test = np_utils.to_categorical(y_test, classes)

        return {'x_train': x_train, 'x_test':x_test ,'y_train':y_train, 'y_test': y_test }


    def process_image(self, image):
        mask = image > 0
        image = image[np.ix_(mask.any(1),mask.any(0))]  
        return cv2.resize(image,(28,28), interpolation = cv2.INTER_LINEAR)

    def build_network(self):
        """Kreira neuronsku mrežu."""
        self.model = Sequential()
        self.model.add(Dense(512,input_dim=784))
        self.model.add(Activation('relu'))                            
        self.model.add(Dropout(0.2))

        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

    def train(self, x_train, x_test, y_train, y_test):
        """Trenira neuronsku mrežu."""
        self.build_network()
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        history = self.model.fit(x_train, y_train, 
          batch_size=128, epochs=20,
          verbose=2,
          validation_data=(x_test, y_test))

    def save_model(self):
        """Čuva model kao json fajl i čuva težine u h5 fajlu."""
        #sačuvaj model kao json
        model_json = self.model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        #sačuvaj težine
        self.model.save_weights("model/model.h5")
        print("Saved model to disk")

    def predict(self, images):
        """Vrši procenu koji broj se nalazi na slici za niz slika. Vraća listu verovatnoća za svaku sliku."""
        network_input = []
        for image in images:
            network_input.append(image.flatten())
        return self.model.predict(np.array(network_input, np.float32))