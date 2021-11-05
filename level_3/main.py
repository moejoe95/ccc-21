import os

import numpy as np
import tensorflow.keras.losses as losses
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dropout
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Sequential

train_file = open(os.path.join('..', 'data', 'level_3', 'train.csv'), 'r')
input_file = open(os.path.join('..', 'data', 'level_3', 'level_3_1.csv'), 'r')

lines = train_file.readlines()

num_samples = int(lines[0].strip())

train_samples = []
train_labels = []

for n in range(num_samples):
    sample_line = lines[1 + n].strip()
    label_line = lines[1 + num_samples + n].strip()
    sample = sample_line.split(',')
    sample = np.asarray([int(x) for x in sample])
    sample = np.reshape(sample, (28, 196))
    label = float(label_line)
    train_samples.append(sample)
    train_labels.append(label)

train_samples = np.asarray(train_samples)
train_labels = np.asarray(train_labels)

X, Y = shuffle(train_samples, train_labels, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

plt.imshow(X_train[0, :, :], cmap='gray')
plt.show()

X_train = X_train / 255.
X_test = X_test / 255.

model = Sequential()
model.add(Input((28, 196, 1)))
model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError(), metrics=[keras.metrics.RootMeanSquaredError()])
model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test))

test_loss, test_acc = model.evaluate(X_test, Y_test, )
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")

model.save("model.hdf5")

lines = input_file.readlines()
num_samples = int(lines[0].strip())
samples = []
for n, line in enumerate(lines[1:]):
    line = line.strip()
    sample = line.split(',')
    sample = [int(x) for x in sample]
    samples.append(sample)

X_test = np.array(samples).reshape(-1, 28, 196, 1)
X_test = X_test / 256.

output_file = "output.csv"
predictions = model.predict(X_test).flatten()

f = open(output_file, "w")
for pred in predictions:
    f.write("{:.4f}\n".format(pred))
f.close()
