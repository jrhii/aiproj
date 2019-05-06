from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def doRun(xTrain, yTrain, xValidate, yValidate, dvName, isBasic = False):
    print('')
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
        model.add(Dense(1, input_dim=inputDim, activation="relu"))
    else:
        #complex
        model.add(Dense(50, input_dim=inputDim, activation="relu"))

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
    model.fit(xTrain, yTrain, epochs=150, batch_size=10)

    scores = model.evaluate(xTrain, yTrain)
    print(model.metrics_names)
    print(scores)
    print("\n%s: %.2f" % (model.metrics_names[1], scores[1]))

    #check against validaiton
    scores = model.evaluate(xValidate, yValidate)
    print("\n%s: %.2f" % (model.metrics_names[1], scores[1]))

    #check what models were predicted
    prediction = model.predict(xValidate)



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

    print(dvName + ' basic run')
    doRun(xTrain, yTrain, xValidate, yValidate, dvName, True)
    print(dvName + ' multi layered run')
    doRun(xTrain, yTrain, xValidate, yValidate, dvName)

print('\a')
# print(model.get_weights())