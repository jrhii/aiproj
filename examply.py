from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def doRun(xTrain, yTrain, xValidate, yValidate, dvName, isBasic = False):
    print('-----------------')
    print('')

    inputDim = len(xValidate[0])

    if not os.path.exists('studentData'):
        os.makedirs('studentData')

    if not os.path.exists('studentData/' + dvName):
        os.makedirs('studentData/' + dvName)

    model = Sequential()

    if (isBasic):
        # this is all for graphing simple relationship
        for col in range(inputDim):
            plt.figure(figsize=(4,4))
            plt.plot(xTrain[:, (col)], yTrain, '.')
            plt.xlabel(str(col))
            plt.ylabel('Grade')
            plt.savefig('studentData/' + dvName + '/Col'+str(col)+'.png')
            plt.close()
            
        #simple fitting
        model.add(Dense(1, input_dim=inputDim, activation="linear"))
    else:
        #complex
        model.add(Dense(50, input_dim=inputDim, activation="linear"))

        # create neural network
        model.add(Dense(35, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='relu'))

    print(model.summary())

    # check for proper neural connections
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])

    #train.
    model.fit(xTrain, yTrain, epochs=150, batch_size=10, verbose=0)

    scores = model.evaluate(xTrain, yTrain)
    print(model.metrics_names)
    print(scores)
    print("\n%s: %.2f" % (model.metrics_names[1], scores[1]))

    #check against validaiton
    scores = model.evaluate(xValidate, yValidate)
    print("\n%s: %.2f" % (model.metrics_names[1], scores[1]))

    #check what models were predicted
    prediction = model.predict(xValidate)


def derivedMathModel(npData):
    weightData = [7.7748566,4.087211,2.8909693,14.525878,7.3228226]
    return weightData[0]*npData[0] + weightData[1]*npData[1] + weightData[2]*npData[2] + weightData[3]*npData[3] + weightData[4]*npData[4] + 34.513798

def derivedReadingModel(npData):
    weightData = [-5.4185405,3.8257022,4.065463,12.958875,12.424449]
    return weightData[0]*npData[0] + weightData[1]*npData[1] + weightData[2]*npData[2] + weightData[3]*npData[3] + weightData[4]*npData[4] + 41.149826

def derivedWritingModel(npData):
    weightData = [-3.0376484,3.8570347,3.7368257,12.563914,10.253032]
    return weightData[0]*npData[0] + weightData[1]*npData[1] + weightData[2]*npData[2] + weightData[3]*npData[3] + weightData[4]*npData[4] + 39.72748

def doDerivedModelRun(xValidate, yValidate, derivedModel):
    cumulativeAbsoluteError = 0
    cumulativeSquaredError = 0

    for index in range(len(xValidate)):
        estimatedValue = derivedModel(xValidate[index])
        absoluteError = abs(estimatedValue - yValidate[index])
        cumulativeAbsoluteError += absoluteError
        cumulativeSquaredError += absoluteError * absoluteError

    meanAbsoluteError = cumulativeAbsoluteError/len(xValidate)
    meanSquaredError = cumulativeSquaredError/len(xValidate)
    print("Mean Absolute Error: %f" % meanAbsoluteError)
    print("Mean Squared Error: %f" % meanSquaredError)



dataset = np.genfromtxt("./StudentsPerformance.csv", skip_header=1, delimiter=",")
# shuffle this data to make sure data is evenly distributed
np.random.shuffle(dataset)

dataSize = dataset.shape[0]

#run data for each test score
for x in range(3):
    index = x + 5

    dvName = 'Math'
    if index == 6:
        dvName = 'Reading'
    elif index == 7:
        dvName = 'Writing'

    # split data into train and validation
    #x is IV, y is DV
    xTrain = dataset[1:math.floor(dataSize*.8), :5]
    yTrain = dataset[1:math.floor(dataSize*.8), index]
    xValidate = dataset[math.floor(dataSize*.8):, :5]
    yValidate = dataset[math.floor(dataSize*.8):, index]

    print('\n------------------------')
    print(dvName + ' basic run\n')
    doRun(xTrain, yTrain, xValidate, yValidate, dvName, True)
    print('')
    print(dvName + ' multi layered run\n')
    doRun(xTrain, yTrain, xValidate, yValidate, dvName)
    # Do our models we made ourselves from an earlier trained model
    if dvName == 'Math':
        print("\nMath Score Derived Model Run")
        doDerivedModelRun(xValidate, yValidate, derivedMathModel)
    elif dvName == 'Reading':
        print("\nReading Score Derived Model Run")
        doDerivedModelRun(xValidate, yValidate, derivedReadingModel)
    elif dvName == 'Writing':
        print("\nWriting Score Derived Model Run")
        doDerivedModelRun(xValidate, yValidate, derivedWritingModel)