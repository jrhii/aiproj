from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import os


bulkDataset = np.genfromtxt("./creditcard.csv", delimiter=",")

print('bulk data')
print('-----------------')
print(bulkDataset.shape)
print('')

# shuffle this data to make sure data is evenly distributed
np.random.shuffle(bulkDataset)

dataSize = bulkDataset.shape[0]

# split data into train and validation
#x is IV, y is DV
xTrain = bulkDataset[0:math.floor(dataSize*.7), :30]
yTrain = bulkDataset[0:math.floor(dataSize*.7), 30]
xValidate = bulkDataset[math.floor(dataSize*.7):, :30]
yValidate = bulkDataset[math.floor(dataSize*.7):, 30]

inputDim = len(xValidate[0])

if not os.path.exists('allData'):
    os.makedirs('allData')

# this is all for graphing simple relationship
for col in range(inputDim):
    plt.figure(figsize=(4,4))
    plt.plot(xTrain[:, (col)], yTrain, '.')
    plt.xlabel(str(col))
    plt.ylabel('Fraudulent')
    plt.savefig('allData/Col'+str(col)+'.png')
    plt.close()

#simple fitting
model = Sequential()
model.add(Dense(1, input_dim=inputDim, activation='relu'))

# create neural network
# model = Sequential()
# model.add(Dense(inputDim*2, input_dim=inputDim, activation='relu'))
# model.add(Dense(inputDim, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
print(model.summary())

# check for proper neural connections
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#train.  We are doing a big batch size here because we want to make sure we get some fraud data
model.fit(xTrain, yTrain, epochs=350, batch_size=10000)

scores = model.evaluate(xTrain, yTrain)
print(model.metrics_names)
print(scores)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#check against validaiton
scores = model.evaluate(xValidate, yValidate)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#check what models were predicted
prediction = model.predict(xValidate)


#import less nonFroud values for a more balancedataset
fraudData = bulkDataset[bulkDataset[:, -1] == 1]
fraudDataLength = len(fraudData)
validData = bulkDataset[bulkDataset[:, -1] == 0]


#1%
def balancedRun(percentage):
    newValidDateLength = int(fraudDataLength/percentage)
    #shuffle random data set to try to get consistently random valid data
    np.random.shuffle(validData)
    balancedDataset = np.concatenate([fraudData, validData[:newValidDateLength]])

    print(str(percentage) + "fraud data ")
    print('-----------------')
    print(balancedDataset.shape)
    print('')

    # shuffle this data to make sure data is evenly distributed
    np.random.shuffle(balancedDataset)

    dataSize = balancedDataset.shape[0]

    # split data into train and validation
    #x is IV, y is DV
    trainPercent = .8 if dataSize > 1000 else .7

    xTrain = balancedDataset[0:math.floor(dataSize*trainPercent), :30]
    yTrain = balancedDataset[0:math.floor(dataSize*trainPercent), 30]
    xValidate = balancedDataset[math.floor(dataSize*trainPercent):, :30]
    yValidate = balancedDataset[math.floor(dataSize*trainPercent):, 30]

    inputDim = len(xValidate[0])

    if not os.path.exists(str(percentage)):
        os.makedirs(str(percentage))

    # this is all for graphing simple relationship
    for col in range(inputDim):
        plt.figure(figsize=(4,4))
        plt.plot(xTrain[:, (col)], yTrain, '.')
        plt.xlabel(str(col))
        plt.ylabel('Fraudulent')
        plt.savefig(str(percentage) + '/DataCol'+str(col)+'.png')
        plt.close()

    #simple fitting
    model = Sequential()
    model.add(Dense(1, input_dim=inputDim, activation='relu'))

    # create neural network
    # model = Sequential()
    # model.add(Dense(inputDim*2, input_dim=inputDim, activation='relu'))
    # model.add(Dense(inputDim, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

    # check for proper neural connections
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #train
    model.fit(xTrain, yTrain, epochs=350, batch_size=1000)

    scores = model.evaluate(xTrain, yTrain)
    print(model.metrics_names)
    print(scores)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

balancedRun(.01)
balancedRun(.1)
balancedRun(1)