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

s = []
a = []
h = []
i = []
with open("Hypertension.csv") as csvFile :
	reader2=csv.reader(csvFile, delimiter = ";")
	for row in reader2 :
		i.append(row[0])
		a.append(row[1])
		s.append(row[2])
		h.append(row[3])
	id2 = [int(n) for n in i[1:]]
	ag = [float(n) for n in a[1:]]
	s = s[1:]
	h = h[1:]

csvFile.close()

######################## Preparation des donnees #########################################

### met les sexes, l'hypertension et l'age dans le meme ordre que les spectres ###
sex = []
age = []
hyp = []
i=0
while i < len(id) :
	n=0
	while n < len(id2) :
		if id2[n] == id[i]:
			if (h[n] == "Y") or (h[n] == "N"):
				if s[n] == "M":
					sex.append(0)
				if s[n] == "F":
					sex.append(1)
				if h[n] == "N":
					hyp.append(0)
				if h[n] == "Y":
					hyp.append(1)
				age.append(ag[n])
			else :
				del rows[i]
		n += 1
	i += 1

###################### Visualisation ###################################################



#plt.plot(ppm, )

### trouve le maximum pour normaliser les donnees ###
m=[]
for n in rows :
    m.append(max(n))
max = max(m)

### normalisation des donnees ###
spectro = []
se = []

i = 0
while i < len(rows) :
    for n in rows[i] :
        se.append(n/max)
    spectro.append(se)
    se = []
    i += 1

### Mettre l'age et le sexe avec les spectres ###
data = [] # sera une liste de listes des donnees des spectres + age + sexe
d = []

i = 0
while i < len(spectro) :
	spectro[i].append(age[i])
	spectro[i].append(sex[i])
	data.append(spectro[i])
	i += 1

# probleme : pas assez de gens en hypertension, reseau n'apprend pas
######################### Creation d'une banque de donnees egale #######################

# 105 hypertension sur 970
# prendre une banque de donnee de 210 patients aleatoire avec les 105

data2 = []
hyp2 = []

i = 0
while i < len(hyp) :
	if hyp[i] == 1:
		hyp2.append(hyp[i])
		data2.append(data[i])
		del hyp[i]
		del data[i]
	else :
		i += 1

n = random.sample(range(0,len(data)), 210-105)

for i in n :
	data2.append(data[i])
	hyp2.append(hyp[i])

hyp2 = np.array(hyp2)
data2 = np.array(data2)

data2 = np.expand_dims(data2, axis=2)

t = random.sample(range(0,len(data2)), 105) # 90% pour le training set

e = list(range(0,len(data2)))
for i in t:
	for j in t:
		if j == i:
			e.remove(j)

v = random.sample(t, 10) # 10% du training set pour la validation set

for i in v:
	for j in t:
		if j == i:
			t.remove(j)

train_data = data2[t]
train_hyp = hyp2[t]

val_data = data2[v]
val_hyp = hyp2[v]

test_data = data2[e]
test_hyp = hyp2[e]

########################## Selection aleatoire sur toutes les donnees #############################
# print(data[0])
# data = np.array(data)
# hyp = np.array(hyp)

# t = random.sample(range(0,len(data)), 873) # 90% pour le training set

# e = list(range(0,len(data)))
# for i in t:
# 	for j in t:
# 		if j == i:
# 			e.remove(j)

# v = random.sample(t, 87) # 10% du training set pour la validation set

# for i in v:
# 	for j in t:
# 		if j == i:
# 			t.remove(j)

# train_data = data[t]
# train_hyp = hyp[t]

# val_data = data[v]
# val_hyp = hyp[v]

# test_data = data[e]
# test_hyp = hyp[e]

############################### Creation du reseau ####################################

model = keras.Sequential()
model.add(keras.layers.Dense(20, activation=tf.nn.relu, input_shape = (1644,)))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

########################## Entrainement du reseau #####################################

history = model.fit(train_data,
                    train_hyp,
                    epochs=50,
                    batch_size=10,
                    validation_data=(val_data, val_hyp),
                    verbose = 1)

######################### Evaluation du reseau #######################################

print("\n")
print("Evaluation :")
results = model.evaluate(test_data, test_hyp)

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









