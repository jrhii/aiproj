from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

dataset = np.genfromtxt("./winequalitytest.csv", delimiter=",")

print('')
print(dataset.shape)
print('')

# shuffle this data to make sure red adn white is evenly distributed
np.random.shuffle(dataset)

# split data into train and validation
#x is IV, y is DV
xTrain = dataset[0:5000, 1:12]
yTrain = dataset[0:5000, 12]
xValidate = dataset[5000:, 1:12]
yValidate = dataset[5000:, 12]

# this is all for graphing simple relationship
for col in range(12):
    plt.figure(figsize=(4,4))
    plt.plot(xTrain[:, (col-1)], yTrain, '.')
    plt.xlabel(str(col - 1))
    plt.ylabel('DV')
    plt.savefig('winetest'+str(col-1)+'.png')

# create neural network
model = Sequential()
model.add(Dense(66, input_dim=11, activation='relu'))
model.add(Dense(44, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# print(model.summary())

# check for proper neural connections
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#train
model.fit(xTrain, yTrain, epochs=150, batch_size=128)

scores = model.evaluate(xTrain, yTrain)
print(model.metrics_names)
print(scores)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#check against validaiton
scores = model.evaluate(xValidate, yValidate)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#check what models were predicted
prediction = model.predict(xValidate)
