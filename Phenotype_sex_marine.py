from __future__ import absolute_import, division, print_function
import math
import csv
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

rows=[]
id=[]
with open("colaus1.focus.raw.csv", "r") as csvFile:
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
with open("sex.csv", "r") as csvFile:
    reader2 = csv.reader(csvFile)
    for row in reader2:
    	s.append(row[1])
    	sex = [int(n) for n in s[1:]]
    	i.append(row[0])
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

train_spectro = np.array(spectro[0:850])
train_sex = np.array(sexe[0:850])

val_spectro = np.array(spectro[851:870])
val_sex = np.array(sexe[851:870])

test_spectro = np.array(spectro[871:])
test_sex = np.array(sexe[871:])

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
                    epochs=40,
                    batch_size=16,
                    validation_data=(val_spectro, val_sex),
                    verbose = 1)


################## Evaluation du modele #################################################
print("\n")
print("Evaluation :")
results = model.evaluate(test_spectro, test_sex)








