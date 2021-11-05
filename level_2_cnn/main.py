import os

import numpy as np
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
from keras.layers import Conv2D, AveragePooling2D
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

train_file = open(os.path.join('..', 'data', 'level_2', 'train.csv'), 'r')

lines = train_file.readlines()

num_samples = int(lines[0].strip())

train_samples = []
train_labels = []

for n in range(num_samples):
    sample_line = lines[1 + n].strip()
    label_line = lines[1 + num_samples + n].strip()
    sample = sample_line.split(',')
    sample = np.asarray([int(x) for x in sample])
    sample = np.reshape(sample, (28, 84))
    label = int(label_line)
    train_samples.append(sample)
    train_labels.append(label)

train_samples = np.asarray(train_samples)
train_labels = np.asarray(train_labels)

X, Y = shuffle(train_samples, train_labels, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


plt.imshow(X_train[0,:,:], cmap='gray')
plt.show()

X_train = X_train / 128.
X_test = X_test / 128.
Y_train = to_categorical(Y_train, 2)
Y_test = to_categorical(Y_test, 2)

# model = Sequential()
# model.add(Input((2352,)))
# model.add(Flatten())
# model.add(Dense(500, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(2, activation='softmax'))
# model.summary()

model = Sequential()
model.add(Input((28,84,1)))
model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=2, activation = 'softmax'))

model.compile(optimizer=optimizers.Adam(), loss=losses.CategoricalCrossentropy(), metrics=['acc'])

model.fit(X_train, Y_train, epochs=50)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")
