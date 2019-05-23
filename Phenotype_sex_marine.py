from __future__ import absolute_import, division, print_function
import math
import csv
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import random

rows=[]
id=[]
datadir = "../../data/"
with open(datadir+"colaus1.focus.raw.csv", "r") as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
    	r = [float(n) for n in row]
    	rows.append(r[1:])
    	id.append(int(r[0]))

csvFile.close()
ppm = rows[0]
rows = rows[1:]
id = id[1:] # enleve le 0 inutile

# graphique pour visualiser

# plt.plot(rows[0], rows[15])
# plt.title(id[15])
# plt.xlabel("ppm")
# plt.axis(ymax = 2500000, ymin = 0, xmin=8.51, xmax=0.8)

# plt.show()

s = []
i = []
with open(datadir+"sex.csv", "r") as csvFile:
    reader2 = csv.reader(csvFile)
    for row in reader2:
    	s.append(row[1])
    	i.append(row[0])
    sex = [int(n) for n in s[1:]]
    id2 = [int(n) for n in i[1:]]

csvFile.close()

######################## Preparation des donnees #########################################
# diviser les donnees par le max --> entre 0 et 1
sexe = []
i=0
while i < len(id) :
    n=0
    while n < len(id2) :
        if id2[n] == id[i]:
            sexe.append(sex[n])
        n += 1
    i += 1

m=[]
for n in rows :
    m.append(max(n))

sexe=np.array(sexe)

spectro = []
s = []
max = max(m)

i = 0
while i < len(rows) :
    for n in rows[i] :
        s.append(n/max)
    spectro.append(s)
    s = []
    i += 1

spectro = np.array(spectro)
sexe = np.array(sexe)

# randomisation des echantillons
# utiliser la fonction numpy.choice(liste, combien on en veut, replace =F)

t = random.sample(range(0,len(spectro)), 874) # 90% pour le training set

e = list(range(0,len(spectro)))
for i in t:
	for j in t:
		if j == i:
			e.remove(j)

v = random.sample(t, 88) # 10% du training set pour la validation set

for i in v:
	for j in t:
		if j == i:
			t.remove(j)

train_spectro = spectro[t]
train_sex = sexe[t]

val_spectro = spectro[v]
val_sex = sexe[v]

test_spectro = spectro[e]
test_sex = sexe[e]

################## Creation du modele ###################################################

model = keras.Sequential()
model.add(keras.layers.Dense(80, activation=tf.nn.relu, input_shape=(1642,)))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

################## Compilation du modele avant de l'entrainer ###########################

model.compile(optimizer='adam', # Comment le modele est telecharges de la base it sees and its loss function.
              loss='binary_crossentropy', # mesure la precision du modele pendant l'entrainement, on veut minimiser pour qu'il aille dans la bonne direction
              metrics=['accuracy'])
# accuracy: the fraction of the images that are correctly classified

history = model.fit(train_spectro,
                    train_sex,
                    epochs=60,
                    batch_size=10,
                    validation_data=(val_spectro, val_sex),
                    verbose = 1)

################## Evaluation du modele #################################################
print("\n")
print("Evaluation :")
results = model.evaluate(test_spectro, test_sex)

history_dict = history.history
history_dict.keys()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss', color ="blue")
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss', color ="blue")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc', color ="blue")
plt.plot(epochs, val_acc, 'b', label='Validation acc', color ="blue")
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


