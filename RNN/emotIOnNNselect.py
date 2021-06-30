"""
███████╗███╗   ███╗ ██████╗ ████████╗██╗ ██████╗ ███╗   ██╗    ██████╗     ██████╗
██╔════╝████╗ ████║██╔═══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║    ╚════██╗   ██╔═████╗
█████╗  ██╔████╔██║██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║     █████╔╝   ██║██╔██║
██╔══╝  ██║╚██╔╝██║██║   ██║   ██║   ██║██║   ██║██║╚██╗██║    ██╔═══╝    ████╔╝██║
███████╗██║ ╚═╝ ██║╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║    ███████╗██╗╚██████╔╝
╚══════╝╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚══════╝╚═╝ ╚═════╝

emotIOn 2.0
By: Andres Rico - Visitng Researcher @ MIT Media Lab's City Science Group.

This file contains the main functions needed to conduct experiments with the emotIOn project. The file contains a Neural Network that can
be trained for individual users with training data obtained by using the emotIOn app, available from the Google Play Store.

This file can also be used for online predictions with trained models. Uncomment sections to activate fucntionalities.

"""


import tensorflow as tf                                                         #Import needed libraries
from tensorflow import keras                                                    #Machine learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM #, CuDNNLSTM    #add CuDNNLSTM to run in GPU
import numpy as np                                                              #Array handling
import matplotlib.pyplot as plt                                                 #Plotting
import socket                                                                   #UDP Communication
import time
import re
from customFunc import inputRawData

#this section is only used for live predicting sessions. UDP communication with terMITe is established.
#UDP_IP = "192.168.0.23" #Specify IP Address for communication.
#UDP_PORT = 19990 #Specify UDP Communication port

#sock = socket.socket(socket.AF_INET, # Internet
#                     socket.SOCK_DGRAM) # UDP
#sock.bind((UDP_IP, UDP_PORT))


## TRAINING DATA INPUT
a = inputRawData("TRAINING DATA SELECTION:", "training matrix")

## TEST DATA INPUT
b = inputRawData("TEST DATA SELECTION:", "testing matrix")

'''
for i in range(50): #Range 50
    np.random.shuffle(a) #Shuffle Data Set
    np.random.shuffle(b) #Shuffle Data Set
'''
#index = int(round(a.shape[0] * .7)) #Divide set into training (70%) and test (30%)

## TRAINING VECTORS
X = a[:,0:6] #Create X atrix for training.
Y = a[:,10] #Create Y label vector for training.
Y = Y - 1 #Adjust Y label vector values to fit NN.

## TEST VECTORS
Xtest = b[:,0:6] #Create X matrix for testing.
Ytest = b[:,10] #Create Y vector for testing.
Ytest = Ytest - 1 #Adjust values of Y vector.

class_names = ['N', 'HH', 'LH', 'LL', 'HL'] #label to know different classes of affetive states with$

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
Xtest = np.reshape(Xtest, (Xtest.shape[0], 1, Xtest.shape[1]))

#Begin sequential model.
model = Sequential()

#First layer of model LSTM. input shape expects input of the size of each X instance.

model.add(LSTM(256, input_shape=(1, X.shape[2]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(256, activation = 'relu')) #Uncomment to run on CPU
model.add(Dropout(0.2))

#Feeds LSTM results into Dense layers for classification.
model.add(Dense(500, activation='sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(500, activation='sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(5, activation='softmax')) #Only One output Unit

#Declare optimizing fucntion and parameters.
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

#Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

#Fit model and store into history variable.
history = model.fit(X, Y, epochs=300,  batch_size = 128, validation_data=(Xtest, Ytest))

print(history.history.keys()) #terminal outout of accuracy results.

test_loss, test_acc = model.evaluate(Xtest, Ytest) #Evaluate model with test sets (X and Y).

print('Test accuracy:', test_acc) #Terminal print of final accuracy of model.


predictions = model.predict(Xtest) #Uses test set to predict.

model.summary()
model.get_config()
print ('Number of Training Examples Used: ' , Y.size) #Helps get number of training examples used.
print ('Hours of Data;' , (Y.size * 1.5) / 3600) #Calculates hours of data. Intervals of 1.5 seconds are used to obtain data.

#plt.style.use('dark_background')

#Complete sript for plotting end results for accuracy on test and training set across different epochs.
plt.rcParams.update({'font.size': 25})
plt.figure(1)
plt.plot(history.history['acc'], '-') #Plot Accuracy Curve
plt.plot(history.history['val_acc'], ':')
plt.title('Model Accuracy U3')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Test Set'], loc='lower right')
plt.show()


#The following functions are for plotting different resuts from the model.

plt.figure(2)
plt.plot(history.history['loss']) #Plot Loss Curvecompletedata = []
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test Set'], loc='upper left')
plt.show()

plt.figure(3)
for times in range(100):
    if np.argmax(predictions[times]) == Ytest[times]:
        plt.plot(times, (Ytest[times]), 'go')
    else:
        plt.plot(times, np.argmax(predictions[times]), 'rx')
        plt.plot(times, ((Ytest[times])), 'bo')
    plt.axis([0, 100, -1, 5])
    plt.title('Prediction Space')
    plt.legend()
plt.show()

'''
while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    indata = np.fromstring(data, dtype = float, sep = ',')
    indata = indata[0:5]
    indata = np.expand_dims(indata, 0)
    prediction = model.predict(indata)
    print class_names[np.argmax(prediction[0])]
    time.sleep(60)
'''

