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
        model.add(Dense(35, activation='linear'))
        model.add(Dense(20, activation='linear'))
        model.add(Dense(10, activation='linear'))
        model.add(Dense(5, activation='linear'))
        model.add(Dense(1, activation='linear'))

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

    plt.figure(figsize=(4,4))
    plt.plot(yValidate, prediction, '.')
    plt.xlabel('Correct values')
    plt.ylabel('Predictions')
    plt.savefig(dvName+'predictions')


def derivedMathModel(npData):
    weightData = [2.6906877,-1.5936518,0.5222445,2.9739928,3.789481,-0.97764665,3.7083657,4.6513343,2.249893,-7.829781,5.7324038,2.6812954]

    prediction = 62.48143
    for index in range(len(weightData)):
        prediction += weightData[index]*npData[index]

    return prediction

def derivedReadingModel(npData):
    weightData = [-3.3387806,-0.9136137,2.2879055,3.6926124,0.07265424,-0.8841467,3.6134818,5.0034957,2.10664,-8.865663,3.908539,3.569657]

    prediction = 65.25066
    for index in range(len(weightData)):
        prediction += weightData[index]*npData[index]

    return prediction

def derivedWritingModel(npData):
    weightData = [-4.333905,-1.4387798,1.8047222,5.002908,-0.30829197,-1.83069,3.7246437,4.6531396,3.6077342,-7.578493,4.3872924,4.922157]

    prediction = 64.64882
    for index in range(len(weightData)):
        prediction += weightData[index]*npData[index]

    return prediction

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
    index = x + 12

    dvName = 'Math'
    if index == 13:
        dvName = 'Reading'
    elif index == 14:
        dvName = 'Writing'

    # split data into train and validation
    #x is IV, y is DV
    xTrain = dataset[1:math.floor(dataSize*.8), :12]
    yTrain = dataset[1:math.floor(dataSize*.8), index]
    xValidate = dataset[math.floor(dataSize*.8):, :12]
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