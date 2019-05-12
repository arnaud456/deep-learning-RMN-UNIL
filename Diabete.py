import csv
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random

######################### Chargement des donnees ########################################

rows=[]
with open("spectres_age_sex.csv", "r") as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
    	rows.append(row)
csvFile.close()

ppm = [float(n) for n in rows[0][1:1643]]
del rows[0]

id = []
dat = []

for r in rows:
	i = int(r[0])
	d = [float(n) for n in r[1:]]
	id.append(i)
	dat.append(d)

d = []
i = []
with open("Phenotypes.csv") as csvFile :
	reader=csv.reader(csvFile, delimiter = ";")
	for row in reader :
		i.append(row[0])
		d.append(row[4])

csvFile.close()

del i[0]
del d[0]

id2 = []
db = []
n = 0
while n < len(d):
	if d[n] != "NA":
		id2.append(int(i[n]))
		db.append(int(d[n]))
	n +=1

######################## Preparation des donnees #########################################

diabete = []
data = []
i=0
while i < len(id) :
	n=0
	while n < len(id2) :
		if id2[n] == id[i]:
			diabete.append(db[n])
			data.append(dat[i])
		n += 1
	i += 1

######################### Creation d'une banque de donnees egale #######################

# 52 diabete sur 940
# prendre une banque de donnee de 52 patients aleatoire avec les 52

data2 = []
diabete2 = []

i = 0
while i < len(diabete) :
	if diabete[i] == 1:
		diabete2.append(diabete[i])
		data2.append(data[i])
		del diabete[i]
		del data[i]
	else :
		i += 1

n = random.sample(range(0,len(data)), 52)

for i in n :
	data2.append(data[i])
	diabete2.append(diabete[i])

diabete2 = np.array(diabete2)
data2 = np.array(data2)

t = random.sample(range(0,len(data2)), 93) # 90% pour le training set

e = list(range(0,len(data2)))
for i in t:
	for j in t:
		if j == i:
			e.remove(j)

v = random.sample(t, 9) # 10% du training set pour la validation set

for i in v:
	for j in t:
		if j == i:
			t.remove(j)

train_data = data2[t]
train_diabete = diabete2[t]

val_data = data2[v]
val_diabete = diabete2[v]

test_data = data2[e]
test_diabete = diabete2[e]

########################## Selection aleatoire sur toutes les donnees #############################

# data = np.array(data)
# diabete = np.array(diabete)

# t = random.sample(range(0,len(data)), 846) # 90% pour le training set

# e = list(range(0,len(data)))
# for i in t:
# 	for j in t:
# 		if j == i:
# 			e.remove(j)

# v = random.sample(t, 84) # 10% du training set pour la validation set

# for i in v:
# 	for j in t:
# 		if j == i:
# 			t.remove(j)

# train_data = data[t]
# train_diabete = diabete[t]

# val_data = data[v]
# val_diabete = diabete[v]

# test_data = data[e]
# test_diabete = diabete[e]

############################### Creation du reseau ####################################

model = keras.Sequential()
model.add(keras.layers.Dense(60, activation=tf.nn.relu, input_shape=(1644,)))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

########################## Entrainement du reseau #####################################

history = model.fit(train_data,
                    train_diabete,
                    epochs=40,
                    batch_size=20,
                    validation_data=(val_data, val_diabete),
                    verbose = 1)

######################### Evaluation du reseau #######################################

print("\n")
print("Evaluation :")
results = model.evaluate(test_data, test_diabete)

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










