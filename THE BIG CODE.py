# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:44:23 2019

@author: THEO
"""

######THE BIG CODE######

############Créer le dataFrame du bonheur#####################

import csv

#df de phenotypes

with open('C:/Users/THEO/Desktop/Biologie-IV/Programmation/data/metab.tsv', 'r') as csvAge:
    id = []
    age = []
    sex = []
    dbtld = []
    bpd = []
    bps = []
    bdmsix = []
    hdlch = []
    ldlch =[]
    trig = []
    readerPhen = csv.reader(csvAge, delimiter="\t")
    next(readerPhen)
    i=0
    for row in readerPhen:
       
        if('' not in row and 'K' not in row): #pour enlever les lignes qui ne nous intéressent pas
            id.append(int(row[0]))
            age.append(float(row[1]))
            sex.append(row[2])
            dbtld.append(row[3])
            bpd.append(int(row[5]) - int(row[4]))
            bps.append(int(row[7]) - int(row[6]))
            bdmsix.append(int(row[7]))
            hdlch.append(float(row[8]))
            ldlch.append(float(row[9]))
            trig.append(float(row[10]))

csvAge.close()

#normalisation des données du phenotype#

def normalisator(tableau):
    maxTableau = max(tableau)
    tableauNormalise = [i/maxTableau for i in tableau]
    return tableauNormalise

import pandas

dfPhen = pandas.DataFrame({'key': id, 'age': normalisator(age), 'dbtld' : dbtld, 'bpd' : normalisator(bpd),'bps' : normalisator(bps), 'bdmsix' : normalisator(bdmsix), 'hdlch' : normalisator(hdlch), 'ldlch': normalisator(ldlch), 'trig': normalisator(trig)})

#df de spectres

with open("C:/Users/THEO/Desktop/Biologie-IV/Programmation/data/colaus1.focus.raw.csv", "r") as csvFile:
    rowSpectre=[]
    idSpectre=[]
    readerSpectre = csv.reader(csvFile)
    next(readerSpectre)
    for row in readerSpectre:
    	r = [float(n) for n in row]
    	rowSpectre.append([n for n in r[1:]])
    	idSpectre.append(int(r[0]))

csvFile.close()

#normalisation des données pour Spectre###

maxDeChaqueSpectres = []
for i in rowSpectre :
    maxDeChaqueSpectres.append(max(i))
maxDeTousLesSpectres = max(maxDeChaqueSpectres)

i=0
while i < len(rowSpectre):
    rowSpectre[i] = [n/maxDeTousLesSpectres for n in rowSpectre[i]]
    i= i+1

dfSpectrum = pandas.DataFrame({'key': idSpectre, 'spectre': rowSpectre})

#df du merge des deux

dfFinal = pandas.merge(dfPhen,dfSpectrum, on ='key')

####################Randomiser les données#################################

import random

idTest = random.sample(range(len(dfFinal)), int(len(dfFinal)*0.9))

#dataFrame pour l'apprentissage du réseau
dfTest = dfFinal.iloc[idTest]
print(dfTest)

#dataFrame pour évaluer le réseau
idEval = []
for i in range(1, len(dfFinal)):
    if(i not in idTest):
        idEval.append(i)
dfEval = dfFinal.iloc[idEval]

####################Réseau de neurone trop cool#########################

#tableaux pour test
listSpectreTest = []
listAgeTest =[]
listSexTest=[]
listSpectreTest = list(dfTest['spectre'])
listAgeTest = list(dfTest['age'])
listSexTest = list(dfTest['sex'])
dataTest = []
i = 0
while i < len(listSpectreTest) :
    listSpectreTest[i].append(listAgeTest[i])
    if listSexTest[i] is 'F':
        listSpectreTest[i].append(0)
    else :
        listSpectreTest[i].append(1)
    dataTest.append(listSpectreTest[i])
    i += 1
print(dataTest[5])

reponseTest = []
i= 0
while i < len(dfTest):
    dbtldNumber = 1
    if list(dfTest['dbtld'])[i] is 'N':
        dbtldNumber = 0
    ligne = [list(dfTest['hdlch'])[i],list(dfTest['bpd'])[i],list(dfTest['bps'])[i],list(dfTest['bdmsix'])[i],list(dfTest['ldlch'])[i],list(dfTest['trig'])[i]]
    reponseTest.append(ligne)
    i=i+1
print(reponseTest[5])

#tableau pour validation

listSpectreEval = []
listAgeEval =[]
listSexEval=[]
listSpectreEval = list(dfEval['spectre'])
listAgeEval = list(dfEval['age'])
listSexEval = list(dfEval['sex'])
dataEval = []
i = 0
while i < len(listSpectreEval) :
    listSpectreEval[i].append(listAgeEval[i])
    if listSexEval[i] is 'F':
        listSpectreEval[i].append(0)
    else :
        listSpectreEval[i].append(1)
    dataEval.append(listSpectreEval[i])
    i += 1
print(dataEval[5])

reponseEval = []
i= 0
while i < len(dfEval):
    ligne = [list(dfEval['hdlch'])[i],list(dfEval['bpd'])[i],list(dfEval['bps'])[i]]
    reponseEval.append(ligne)
    i=i+1
print(reponseEval[5])

####################Réseau de neurone trop cool#########################

import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.Dense(80, activation=tf.nn.relu, input_shape=(1644,)))
model.add(keras.layers.Dense(3, activation=tf.nn.relu))

model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics=['mean_squared_error'])

history = model.fit(np.array(dataTest),
                    np.array(reponseTest),
                    epochs=60,
                    batch_size=10,
                    validation_data=(np.array(dataEval), np.array(reponseEval)),
                    verbose = 1)

model.summary()

import matplotlib.pyplot as plt

prediction = model.predict(np.array(dataEval))
plt.plot(np.array(reponseEval)[:,0], prediction[:,0], "b.")
plt.show()
plt.plot(np.array(reponseEval)[:,1], prediction[:,1], "b.")
plt.show()
plt.plot(np.array(reponseEval)[:,2], prediction[:,2], "b.")
plt.show()
print(np.corrcoef(np.array(reponseEval)[:,1], prediction[:,1])[0][1])
#mettre ça au carré
