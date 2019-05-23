
import tensorflow as tf
from tensorflow import keras
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def transfo(val,mmin,mmax):
    return (val-mmin)/(mmax-mmin)

def inv_transfo(val,mmin,mmax):
    return val*(mmax-mmin)+mmin

fname = "../../data/colaus1.focus.raw.csv"
sexfname = "../../data/sex.csv"
phenofname = "../../data/metab.tsv"

mdata = pd.read_csv(fname,sep=",",header=0)
sdata = pd.read_csv(sexfname,sep=",",header=0)
pdata = pd.read_csv(phenofname,sep="\t",header=0)

data = mdata.merge(pdata,left_on="0",right_on="pt")
data["sex"] = (data["sex"]=="M")+0

ppms = np.array(mdata.columns[1:],dtype=float)
ids = mdata.iloc[:,0]
mdata = mdata.iloc[:,1:]

model = keras.Sequential()
#model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(60, activation=tf.nn.relu,input_shape=(len(mdata.columns)+1,)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.relu))
model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mean_absolute_error'])
model.summary()

spectrum = data.drop(["0"]+list(pdata.columns),axis=1)
spectrum = spectrum/spectrum.max(axis=0)
spectrum = pd.concat([spectrum.reset_index(drop=True),data["sex"]],axis=1)
train_idx = np.random.choice(range(len(data)),int(len(data)*0.9),replace=False)
train_data = np.array(spectrum.iloc[train_idx,])
minbmi = data["bdmsix"].min()
maxbmi = data["bdmsix"].max()
output = np.array(transfo(data["bdmsix"],minbmi,maxbmi))
train_labels = output[train_idx]

test_idx = np.setdiff1d(range(len(data)),train_idx)
test_data = np.array(spectrum.iloc[test_idx,])
#test_labels =  np.array(data["sex"][test_idx])
test_labels =  output[test_idx]
    
model.fit(train_data,train_labels,batch_size= 10,epochs=50)

#results = model.evaluate(test_data, test_labels)
pred = model.predict(test_data)
corr = np.corrcoef(np.transpose(pred),np.transpose(test_labels))
plt.plot(inv_transfo(test_labels,minbmi,maxbmi),inv_transfo(pred,minbmi,maxbmi),'.')
plt.plot([18,40],[18,40])
plt.title(corr[0,1]*corr[0,1])
plt.show()

# for layer in model.layers:
#     print(layer.output_shape)
