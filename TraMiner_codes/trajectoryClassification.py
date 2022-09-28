import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix,classification_report
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization, Activation, Dropout, Input
from keras.models import Model, Sequential 
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import glob
import os
import numpy as np
import itertools
import operator
from keras.utils import np_utils
from keras.regularizers import l2
import random
import time
import imageio

#===================Read image labels==========================
labelfilename='section_all.csv'
readlabelfile = open(labelfilename, "r")
labeldata=[] 
finalLabels=[]
filebase1=[]
filebase2=[]
npLabels=[]

for line in readlabelfile:
    Type = line.split(",")
    if (float(Type[1])==4 or float(Type[1])==1 or float(Type[1])==2) :
      labeldata.append(np.array(list(map(float,Type))))
npLabels = np.asarray(labeldata, dtype=np.int16)
print(npLabels.shape)


#===================Load and Read TRAJ and SPEED images==========================
rootSpeed = './trajectory_speed/30s/speed30/img_speed/'
all_files_speed = glob.glob(os.path.join(rootSpeed, '*.png'))
all_files_speed.sort(key=lambda x: int(os.path.splitext(x)[0].split("/")[-1].split("_")[0]))

rootTrajectory = './trajectory_speed/30s/trajectory30/img_trajectory/'
all_files_trajectory = glob.glob(os.path.join(rootTrajectory, '*.png'))
all_files_trajectory.sort(key=lambda x: int(os.path.splitext(x)[0].split("/")[-1].split("_")[0]))

for idx, files in enumerate(all_files_speed):
  imgID = int(files.split("/")[-1].split("_")[0] )
  if (imgID in npLabels[:,0]):
    a = np.where(npLabels[:,0] == imgID)
    filebase1.append(files)
    finalLabels.append(npLabels[int(a[0])])

for idx, files in enumerate(all_files_trajectory):
  imgID = int(files.split("/")[-1].split("_")[0] )
  if (imgID in npLabels[:,0]):
    a = np.where(npLabels[:,0] == imgID)
    filebase2.append(files)
finalLabels = np.asarray(finalLabels, dtype=np.int16)


#===================Convert Labels of different categires in numeric order==========================
temp = {i: j for j, i in enumerate(set(finalLabels[:,2]))} 
finalLabels[:,2] = [temp[i] for i in finalLabels[:,2]] 
finalLabels[:,1]= np.where(finalLabels[:,1] == 4, 0,finalLabels[:,1])

healthy = finalLabels[finalLabels[:,1] == 0]
dementia = finalLabels[finalLabels[:,1] == 1] 
mci = finalLabels[finalLabels[:,1] == 2]

uniqH = np.unique(healthy[:,2])
uniqD = np.unique(dementia[:,2])
uniqM = np.unique(mci[:,2])

persons = np.unique(finalLabels[:,2])
allCuniq=np.unique(finalLabels[:,1])

#===================================================================================
#===================Leave one person out cross validation function==========================
#===================================================================================
def leaveOnePersonOut(filebase,finalLabels,testPersonNum):
  ytest =[]
  xtest =[]
  xtrain =[]
  ytrain =[]
  xValidation=[]
  yValidation=[]
  item = 0
  variance=[]
  totalData=[]
  personData = 0
  val_count = 0
  
  for idx, npfile in enumerate(filebase):
    imgID= npfile.split("/")[-1].split("_")[0]
    img=load_img(npfile)  
    img_array = img_to_array(img)
    np_array =img_array.reshape(-1)     
    totalData.append(np_array)

  totalData = np.array(totalData)
  #totalData= np.stack(totalData)
  print(totalData.shape)
  
  valDem=0

  for i in range(0,80):
     if (uniqH[i] == testPersonNum):
      dataIndex = np.where(finalLabels[:,2] == uniqH[i])
      for index in dataIndex[0]:
         xtest.append(totalData[index])
         ytest.append(int(finalLabels[index,1]))

     if (uniqH[i] != testPersonNum and val_count >=0 and val_count <4):
      valDem=uniqH[i]
      dataIndex = np.where(finalLabels[:,2] == uniqH[i])
      for index in dataIndex[0]:
         xValidation.append(totalData[index])
         yValidation.append(int(finalLabels[index,1]))
      val_count += 1

     if (uniqH[i]!= valDem and uniqH[i] != testPersonNum and val_count >=4 ):
      dataIndex = np.where(finalLabels[:,2] == uniqH[i])
      for index in dataIndex[0]:
         xtrain.append(totalData[index])
         ytrain.append(int(finalLabels[index,1]))

  for j in range(0,19):
    if (uniqD[j] == testPersonNum):
      dataIndex = np.where(finalLabels[:,2] == uniqD[j])
      for index in dataIndex[0]:
         xtest.append(totalData[index])
         ytest.append(int(finalLabels[index,1]))

    if (uniqD[j] != testPersonNum and val_count <6):
      valDem=uniqD[j]
      dataIndex = np.where(finalLabels[:,2] == uniqD[j])
      for index in dataIndex[0]:
         xValidation.append(totalData[index])
         yValidation.append(int(finalLabels[index,1]))
      val_count += 1

    if (uniqD[j]!= valDem and uniqD[j] != testPersonNum and val_count >=5 ):
      dataIndex = np.where(finalLabels[:,2] == uniqD[j])
      for index in dataIndex[0]:
         xtrain.append(totalData[index])
         ytrain.append(int(finalLabels[index,1]))   

  for i in range(0,54):
     if (uniqM[i] == testPersonNum):
      dataIndex = np.where(finalLabels[:,2] == uniqM[i])
      for index in dataIndex[0]:
         xtest.append(totalData[index])
         ytest.append(int(finalLabels[index,1]))

     if (uniqM[i] != testPersonNum and val_count >=0 and val_count <9):
      valDem=uniqM[i]
      dataIndex = np.where(finalLabels[:,2] == uniqM[i])
      for index in dataIndex[0]:
         xValidation.append(totalData[index])
         yValidation.append(int(finalLabels[index,1]))
      val_count += 1

     if (uniqM[i]!= valDem and uniqM[i] != testPersonNum and val_count >=9 ):
      dataIndex = np.where(finalLabels[:,2] == uniqM[i])
      for index in dataIndex[0]:
         xtrain.append(totalData[index])
         ytrain.append(int(finalLabels[index,1]))

  return np.array(xtrain) , np.array(ytrain) , np.array(xtest), np.array(ytest), np.array(xValidation), np.array(yValidation)

#===============================================
#============One input MLP DNN==================
#===============================================
def buildModel1Input(inputDim):
  inputA = Input(shape=inputDim)

  # the first branch operates on the first input
  x = Dense(32, kernel_initializer='he_uniform', kernel_regularizer=l2(0.01))(inputA)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dense(32, kernel_initializer='he_uniform', kernel_regularizer=l2(0.01))(x)
  x = BatchNormalization()(x)
  x = Dropout(0.5)(x)
  x = Activation('relu')(x)
  x = Dense(3)(x)
  x = BatchNormalization()(x)
  x = Activation('softmax')(x)

  model = Model(inputs=inputA, outputs=x)
  opt = Adam(lr=INIT_LR)
  model.compile(loss='categorical_crossentropy', optimizer=opt,metrics = ['acc'])

  return model

#===============================================
#============Two inputs MLP DNN==================
#===============================================
def twoInputsTrajectoryClassification(inputDim):
  # define two sets of inputs
  inputA = Input(shape=inputDim)
  inputB = Input(shape=inputDim)

  # the first branch operates on the first input
  x = Dense(32, kernel_initializer='he_uniform', kernel_regularizer=l2(0.01))(inputA)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dense(32, kernel_initializer='he_uniform', kernel_regularizer=l2(0.01))(x)
  x = BatchNormalization()(x)
  x = Dropout(0.5)(x)
  x = Activation('relu')(x)
  x = Model(inputs=inputA, outputs=x)

  # the second branch opreates on the second input
  y = Dense(32, kernel_initializer='he_uniform', kernel_regularizer=l2(0.01))(inputB)
  y = BatchNormalization()(y)
  y = Activation('relu')(y)
  y = Dense(32, kernel_initializer='he_uniform', kernel_regularizer=l2(0.01))(y)
  y = BatchNormalization()(y)
  y = Dropout(0.5)(y)
  y = Activation('relu')(y)
  y = Model(inputs=inputB, outputs=y)

  # combine the output of the two branches
  combined = concatenate([x.output, y.output])
  #z = Dropout(0.5)(combined)
  z = Dense(3)(combined)
  z = BatchNormalization()(z)
  z = Activation('softmax')(z)
  # our model will accept the inputs of the two branches and
  # then output a single value
  model = Model(inputs=[x.input, y.input], outputs=z)
  opt = Adam(lr=INIT_LR)
  model.compile(loss='categorical_crossentropy', optimizer=opt,metrics = ['acc'])
  return model

#===============================================
#============Gochoo's DCNN======================
#===============================================
def DCNN(xTrain):
  model = Sequential()
  model.add(Conv2D(32, (5, 5), padding='same',input_shape=xTrain.shape[1:]))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Conv2D(128, (5, 5), padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Conv2D(256, (5, 5), padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Flatten())

  model.add(Dense(512))
  model.add(Activation('relu'))


  model.add(Dense(128))
  model.add(Activation('relu'))


  model.add(Dense(64))
  model.add(Activation('relu'))

  model.add(Dense(3))
  model.add(Activation('softmax'))
  opt = Adam(lr=0.0001)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

  return model

#============Initialization===========================
#=====================================================
num_classes = 3
cm=0
width = 100
height = 130
cvscores = []
accscores = []
inputDim=(39000,)
fscoresMicro = []
precisionscoresMicro = []
recallscoresMicro = []
timeElapsedstring =""
timeElapsed =0
conf_matrix_list_of_arrays=0
EPOCHS = 27
BS =64
INIT_LR = 0.00001

lowperson = []
lowpersonH50= []
lowpersonD50= []
lowpersonM50= []
lowpersonH= []
lowpersonD= []
lowpersonM= []

totalprediction=[]
totaltest=[]
longtermAnal=np.zeros((3,3), dtype=np.int16)
equalConf=[]

#===============================================
#============Main Program==================
#===============================================

for i in persons:
  print("Person number:" + str(i)) 
  xTrain1, yTrain1, xTest1, yTest1,xVal1, yVal1 = leaveOnePersonOut(filebase1,finalLabels,i)
  xTrain2, yTrain2, xTest2, yTest2,xVal2, yVal2 = leaveOnePersonOut(filebase2,finalLabels,i)


  xTrain1 = xTrain1.astype('float32')
  xTest1 = xTest1.astype('float32')
  xVal1 = xVal1.astype('float32')	
  xTrain2 = xTrain2.astype('float32')
  xTest2 = xTest2.astype('float32')
  xVal2 = xVal2.astype('float32')

  xTrain1 /= 255.0
  xTest1 /= 255.0
  xVal1 /= 255.0

  xTrain2 /= 255.0
  xTest2 /= 255.0
  xVal2/= 255.0

  yTrain1 =  keras.utils.to_categorical(yTrain1, num_classes)
  yTest1 =  keras.utils.to_categorical(yTest1, num_classes)
  yVal1 =  keras.utils.to_categorical(yVal1, num_classes)
  
  model = twoInputsTrajectoryClassification(inputDim)
  #model = buildModel1Input(inputDim)
  #model = DCNN(xTrain1)

  t0 = time.clock()
  history = model.fit([xTrain1,xTrain2],yTrain1, validation_data =([xVal1,xVal2], yVal1), batch_size = BS, verbose=0, epochs=EPOCHS )
  t1 = time.clock()
  print("Time elapsed (sec): ", t1 - t0) # CPU seconds elapsed (floating point)

  timeElapsedstring =  timeElapsedstring + "Time elapsed (sec): " + str(t1 - t0) + "\n"
  timeElapsed =timeElapsed + (t1 - t0)
  #model.save_weights(filepath + "model" +str(i) + ".h5")
  y_pred = model.predict([xTest1,xTest2])
  #y_pred= np.round(y_pred).reshape(-1)
  yTest1=yTest1.argmax(axis=1)
  y_pred = y_pred.argmax(axis=1)  
  
  conf = confusion_matrix(yTest1,y_pred, labels=[0,2,1])
  conf_matrix_list_of_arrays = conf_matrix_list_of_arrays + conf 


  print("classification_report:\n" , classification_report(yTest1,y_pred))
  print('===============================================================')
  print('Confusion Martrix:')
  print(conf)
  print('===============================================================')

  for t in y_pred:
    totalprediction.append(t)

  for t in yTest1:
    totaltest.append(t)
 
  #===========================Long Term Analysis=========================
  index = np.unravel_index(np.argmax(conf, axis=None), conf.shape)
  equality = np.argwhere(conf==conf[index])
  if (len(equality) < 2):
    longtermAnal[index] = 1+ longtermAnal[index]
  else:
    equalConf.append(i)
  #======================================================================


  accscores.append(accuracy_score(yTest1,y_pred)*100)
  if(accuracy_score(yTest1,y_pred) < 0.5):
    lowperson.append(i)


  if(accuracy_score(yTest1,y_pred) == 0.5 and np.unique(yTest1)==0):
    lowpersonH50.append(i)

  if(accuracy_score(yTest1,y_pred) == 0.5 and np.unique(yTest1)==1):
    lowpersonD50.append(i)

  if(accuracy_score(yTest1,y_pred) == 0.5 and np.unique(yTest1)==2):
    lowpersonM50.append(i)

print('id person with lower than 50:')
print(lowperson)

print('healthy person id with equal to 50:')
print(lowpersonH50)

print('dementia  person id with equal to 50:')
print(lowpersonD50)
print('mci person id with equal to 50:')
print(lowpersonM50)

print('id person with equal max sample recognition:')
print(equalConf)

print("Precision Macro Average:" , precision_score(totaltest,totalprediction, average='macro')*100)

print("Recall Macro Average:" , recall_score(totaltest,totalprediction, average='macro')*100)

print("F1-Score Macro Average:" , f1_score(totaltest,totalprediction, average='macro')*100)


print('Confusion MAtrix of Samples:')
print(conf_matrix_list_of_arrays)

print('Confusion MAtrix of LongTerm Analysis:')
print(longtermAnal)
