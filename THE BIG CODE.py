# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:44:23 2019

@author: THEO
"""

######THE BIG CODE######

############Créer le dataFrame du bonheur#####################

import csv

spectrumfile = "../../data/colaus1.focus.raw.csv"
phenofile = "../../data/metab.tsv" 
#csvAge = "../../data/sex.csv"
#df de phenotypes


with open(phenofile, 'r') as csvAge:
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

msex = [1  if s=='M' else 0 for s in sex]
dfPhen = pandas.DataFrame({'key': id, 'age': normalisator(age), 'dbtld' : dbtld, 'bpd' : normalisator(bpd),'bps' : normalisator(bps), 'bdmsix' : normalisator(bdmsix), 'hdlch' : normalisator(hdlch), 'ldlch': normalisator(ldlch), 'trig': normalisator(trig),'sex': msex})

#df de spectres

with open(spectrumfile, "r") as csvFile:
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

###########fonction pour randomiser################


import random

idTest = random.sample(range(len(dfFinal)), int(len(dfFinal)*0.9))

idEval = []
for i in range(1, len(dfFinal)):
    if(i not in idTest):
        idEval.append(i)
        
dfTest = dfFinal.iloc[idTest]
dfEval = dfFinal.iloc[idEval]

#print(dfTest['spectre'].iloc[0])

def randomData(dataframe) :
    data = []
    for i in range(len(dataframe)):
        listSpectre=[]
        listSpectre = list(dataframe['spectre'].iloc[i])
        listSpectre.append(dataframe['age'].iloc[i])
        if dataframe['sex'].iloc[i] is 'F':
            listSpectre.append(0)
        else :
            listSpectre.append(1)
        data.append(listSpectre)
    return data

def randomReponse(dataframe):
    reponse = []
    for i in range(len(dataframe)):
        dbtldNumber = 1
        if list(dataframe['dbtld'])[i] is 'N':
            dbtldNumber = 0
        ligne = [list(dataframe['hdlch'])[i],list(dataframe['bpd'])[i],list(dataframe['bps'])[i],list(dataframe['bdmsix'])[i],list(dataframe['ldlch'])[i],list(dataframe['trig'])[i], dbtldNumber]
        reponse.append(ligne)
    return reponse

dataTest = randomData(dfTest)
reponseTest = randomReponse(dfTest)

dataEval = randomData(dfEval)
reponseEval = randomReponse(dfEval)


####################Réseau de neurone trop cool#########################

#####calcul des coefficient de correlation pour chaque phénotype

import numpy as np
import tensorflow as tf
from tensorflow import keras

additionCorrCarre = [0,0,0,0,0,0,0]

nbessai = 20
nepoch = 5

model = keras.Sequential()
model.add(keras.layers.Dense(80, activation=tf.nn.relu, input_shape=(1644,)))
model.add(keras.layers.Dense(1, activation=tf.nn.relu))

model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics=['mean_squared_error'])

for i in range(nbessai):
    
    idTest = random.sample(range(len(dfFinal)), int(len(dfFinal)*0.9))
    
    idEval = []
    for i in range(1, len(dfFinal)):
        if(i not in idTest):
            idEval.append(i)
            
    dfTest = dfFinal.iloc[idTest]
    dfEval = dfFinal.iloc[idEval]
    
    dataTest = randomData(dfTest)
    reponseTest = randomReponse(dfTest)
    
    dataEval = randomData(dfEval)
    reponseEval = randomReponse(dfEval)
    

    #je calcul corrcarre pour ces valeurs là
    corrCarre = []
    for i in range(len(reponseEval[0])):
      
        
        model.fit(np.array(dataTest),
                            np.array(reponseTest)[:,1],
                            epochs=nepoch)
        
        prediction = model.predict(np.array(dataEval))
        dataPrediction = []
        for i in prediction:
            dataPrediction.append(i[0])
        corrCarre.append((np.corrcoef(np.array(reponseEval)[:,1], np.array(dataPrediction))[0][1])*(np.corrcoef(np.array(reponseEval)[:,1], np.array(dataPrediction))[0][1]))
    
    #j'additione corrcarre à ses anciennes valeurs dans le tableau additionCorrCarre
    for i in range(len(corrCarre)):
        additionCorrCarre[i] = additionCorrCarre[i]+corrCarre[i]

        
    additionCorrCarre2 = [0,0,0,0,0,0,0]


    
    model = keras.Sequential()
    model.add(keras.layers.Dense(80, activation=tf.nn.relu, input_shape=(1644,)))
    model.add(keras.layers.Dense(7, activation=tf.nn.relu))
    
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error'])
for i in range(nbessai):
    idTest = random.sample(range(len(dfFinal)), int(len(dfFinal)*0.9))
    
    idEval = []
    for i in range(1, len(dfFinal)):
        if(i not in idTest):
            idEval.append(i)
            
    dfTest = dfFinal.iloc[idTest]
    dfEval = dfFinal.iloc[idEval]
    
    dataTest = randomData(dfTest)
    reponseTest = randomReponse(dfTest)
    
    dataEval = randomData(dfEval)
    reponseEval = randomReponse(dfEval)
    
    history = model.fit(np.array(dataTest),
                        np.array(reponseTest),
                        epochs=nepoch, 
                        batch_size=10,
                        validation_data=(np.array(dataEval), np.array(reponseEval)),
                        verbose = 1)
    prediction = model.predict(np.array(dataEval))
    corrCarre2 = [np.corrcoef(np.array(reponseEval)[:,0], prediction[:,0])[0][1],np.corrcoef(np.array(reponseEval)[:,1], prediction[:,1])[0][1],np.corrcoef(np.array(reponseEval)[:,2], prediction[:,2])[0][1],np.corrcoef(np.array(reponseEval)[:,3], prediction[:,3])[0][1],np.corrcoef(np.array(reponseEval)[:,4], prediction[:,4])[0][1],np.corrcoef(np.array(reponseEval)[:,5], prediction[:,5])[0][1],np.corrcoef(np.array(reponseEval)[:,6], prediction[:,6])[0][1]]
    
    for i in range(len(corrCarre2)):
        additionCorrCarre2[i] = additionCorrCarre2[i]+corrCarre2[i]
    
        
#Donc là le premier print devrait afficher les R^2 dans l'ordre:
#hdlch, bpd, bps, dmsix, ldlch, trig, dbtldNumber
#et le deuxième devrait afficher la même chose mais pour un réseau à seule sortie
 
print([n/nbessai for n in additionCorrCarre])
print([n/nbessai for n in additionCorrCarre2])

