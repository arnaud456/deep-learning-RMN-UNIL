import csv
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random

######################### Chargement des spectres ########################################
i = []
a = []
s = []
d = []
rows=[]
with open("spectres_age_sex.csv", "r") as csvFile:
    reader = csv.reader(csvFile, delimiter =";")
    for row in reader:
    	rows.append(row)
    	i.append(row[0])
    	a.append(row[1])
    	s.append(row[2])
    	rows.append(row)

csvFile.close()

ppm = [float(n) for n in rows[0][3:]]
del rows[:2]

id = [int(n) for n in i[1:]]
ag = [float(n) for n in a[1:]]
sex = [int(n) for n in s[1:]]

dat = []
for n in rows:
	da = []
	for m in n:
		da.append(float(m))
	dat.append(da)

########### Hypertension, diabete, hypercholesterolemie, hypertriglyceridemie ##########

d = []
i = []
hte = []
htr = []
hyc = []
with open("Phenotypes.csv") as csvFile :
	reader=csv.reader(csvFile, delimiter = ";")
	for row in reader :
		if (row[3] != "NA") & (row[4] != "NA") & (row[5] != "NA") & (row[6] != "NA") & (row [7] != "NA"):
			i.append(row[0])
			hte.append(row[3])
			d.append(row[4])
			hyc.append(row[6])
			htr.append(row[7])

csvFile.close()

d2 = [int(n) for n in d[1:]]
hte2 = [int(n) for n in hte[1:]]
htr2 = [int(n) for n in htr[1:]]
hyc2 = [int(n) for n in hyc[1:]]
id2 = [int(n) for n in i[1:]]

############################# Preparation des donnees ##################################

# prendre les id qui sont dans le dat

tension = []
choles = []
trygl = []
diabete = []
data = []
age = []
sexe = []

i=0
while i < len(id) :
	n=0
	while n < len(id2) :
		if id2[n] == id[i]:
			age.append(ag[i])
			sexe.append(sex[i])
			diabete.append(d2[n])
			tension.append(hte2[n])
			choles.append(hyc2[n])
			trygl.append(htr2[n])
			data.append(dat[i])
		n += 1
	i += 1

phenotypes = []
i=0
while i < len(tension):
	p = [diabete[i], tension[i], choles[i], trygl[i]]
	phenotypes.append(p)
	i+=1

########################## Selection aleatoire sur toutes les donnees #############################

data = np.array(data)
phenotypes = np.array(phenotypes)

t = random.sample(range(0,len(data)), 845) # 90% pour le training set

e = list(range(0,len(data)))
e = list(set(e)-set(t))

v = random.sample(t, 84) # 10% du training set pour la validation set

for i in v:
	for j in t:
		if j == i:
			t.remove(j)

train_data = data[t]
train_phenotypes = phenotypes[t]

val_data = data[v]
val_phenotypes = phenotypes[v]

test_data = data[e]
test_phenotypes = phenotypes[e]

############################### Creation du reseau ####################################
model = keras.Sequential()
dense1 = keras.layers.Dense(100, activation=tf.nn.relu, input_shape = (1645,))
model.add(dense1)
model.add(keras.layers.Dense(60, activation=tf.nn.relu))
model.add(keras.layers.Dense(4, activation=tf.nn.relu))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

########################## Entrainement du reseau #####################################

history = model.fit(train_data,
                    train_phenotypes,
                    epochs=30,
                    batch_size=12,
                    validation_data=(val_data, val_phenotypes),
                    verbose = 1)

######################### Evaluation du reseau #######################################

print("\n")
print("Evaluation :")
results = model.evaluate(test_data, test_phenotypes)

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

########################### Poids ###################################

weights = dense1.get_weights()

plt.clf()
plt.plot(ppm, abs(weights[0][3:]), "b.")
plt.show()














