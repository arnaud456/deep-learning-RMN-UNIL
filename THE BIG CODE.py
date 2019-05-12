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
       
        if('' not in row): #pour enlever les cases vides
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

print(dfFinal)

####################Randomiser les données#################################

