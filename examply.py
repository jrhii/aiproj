from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import math

dataset = np.genfromtxt("./BlackFriday.csv", skip_header=1, delimiter=",")

print('')
print(dataset.shape)
print('')

# shuffle this data to make sure data is evenly distributed
np.random.shuffle(dataset)

dataSize = dataset.shape[0]

# split data into train and validation
#x is IV, y is DV
xTrain = dataset[0:math.floor(dataSize*.7), 2:8]
yTrain = dataset[0:math.floor(dataSize*.7), 11]
xValidate = dataset[math.floor(dataSize*.7):, 2:8]
yValidate = dataset[math.floor(dataSize*.7):, 11]

# this is all for graphing simple relationship
for col in range(6):
    plt.figure(figsize=(4,4))
    plt.plot(xTrain[:, (col)], yTrain, '.')
    plt.xlabel(str(col))
    plt.ylabel('DV')
    plt.savefig('test'+str(col)+'.png')

# create neural network
model = Sequential()
model.add(Dense(66, input_dim=6, activation='relu'))
model.add(Dense(44, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# print(model.summary())

# check for proper neural connections
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#train
model.fit(xTrain, yTrain, epochs=3, batch_size=128)

scores = model.evaluate(xTrain, yTrain)
print(model.metrics_names)
print(scores)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#check against validaiton
scores = model.evaluate(xValidate, yValidate)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#check what models were predicted
prediction = model.predict(xValidate)
