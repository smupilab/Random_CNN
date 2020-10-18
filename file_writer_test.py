#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-

import subprocess
import math

def fileMaker(gene, index = None, parent = None):

    fitness = gene[0]
    lr = gene[1]
    initW = gene[2]
    optim = gene[3]
    actF = gene[4]
    kernel_size = gene[5]
    depth = gene[6]
    fc_layer = gene[7]
    drop_out = gene[8]
    epoch = gene[9]
    byp = gene[10]

    f = open("created_cnn.py", 'w')

    # import
    f.write("\n")
    f.write('''import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
import copy
import numpy as np
import random
import time
import sys
sys.setrecursionlimit(1000000)\n''')




    #def
    f.write('''
def make_rand(net_list):
  lis=list()
  re_seed=random.randint(1,4) 
  for i in range(re_seed):
    seed=random.randint(1,4) 
    if seed==1:
     im_output= layers.Conv2D(filters=64, kernel_size=[ks,ks], padding='same', activation=actF)(output)
    elif seed==2:
      im_output= layers.Dropout(rate=drop_out)(output)
    elif seed==3:
     im_output= layers.MaxPooling2D(pool_size=[ks, ks], padding='same', strides=1)(output)
    elif seed==4:
     im_output = layers.Activation(actF)(output)
    lis.append(im_output)
  return lis

def make_short_cut(a_layer,b_layer): 
  im_output = layers.Add()([a_layer,b_layer])
  return im_output\n''')

    f.write("start_time = time.time()\n")
    f.write("start_clock = time.clock()\n")

    # gene
    f.write("lr = " + str(lr) + "\n")
    f.write("initW = '" + str(initW) + "'\n")
    if optim == 'Adam':
        f.write("opt = keras.optimizers.Adam(lr =lr, beta_1=0.9, beta_2=0.999, amsgrad=False)\n")
    elif optim == 'Adagrad':
        f.write("opt = keras.optimizers.Adagrad(learning_rate=lr)\n")
    elif optim == 'SGD':
        f.write("opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False)\n")
    elif optim == 'Adadelta':
        f.write("opt = keras.optimizers.Adadelta(learning_rate=lr, rho=0.95)\n")
    f.write("actF = '" + str(actF) + "'\n")
    f.write("ks = " + str(kernel_size) + "\n")
    f.write("depth = " + str(depth) + "\n")
    f.write("fc_layer = " + str(fc_layer) + "\n")
    f.write("drop_out = " + str(drop_out) + "\n")
    f.write("byp = " + str(byp) + "\n")
    f.write('''
img_rows = 28
img_cols = 28

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
(x_train, x_test) = x_train[:30000], x_test[:1000]
(y_train, y_test) = y_train[:30000], y_test[:1000]
input_shape = (img_rows, img_cols, 1)

batch_size = 128
num_classes = 10\n''')
    #f.write("epochs =" + str(epoch) + "\n\n")
    f.write("epochs = 5\n\n")

    f.write("y_train = keras.utils.to_categorical(y_train, num_classes)\n")
    f.write("y_test = keras.utils.to_categorical(y_test, num_classes)\n\n")

    # !
    f.write("inputs = keras.Input(shape = input_shape, name = 'input')\n")
    
    f.write("output = layers.Conv2D(filters = 64, kernel_size = [ks, ks], padding = 'same', activation = actF)(inputs)\n")
    f.write('''
net_list=list()
add_num=0

for depth in range(1): 
  a=make_rand(net_list)
  net_list.extend(a)
  if len(a)==1:r_num=0 
  else:r_num=random.randint(0,len(a)-1)                 
  output=a[r_num]   
  short_cut_dec=random.randint(1,5)     
  if (short_cut_dec==1 or short_cut_dec==2) and len(net_list)>1:
    add_num=add_num+1
    add_layer_num=random.randint(0,len(net_list)-1)
    add_list=[] 
    for _ in range( random.randint(0,len(net_list)-1) ):
      a_layer_num=random.randint(0,len(net_list)-1)
      add_list.append(a_layer_num)
      c=make_short_cut(net_list[a_layer_num],output)
      output=c
    net_list.append(net_list)\n''')





    f.write("output = layers.GlobalAveragePooling2D()(output)\n")
    # Dense
    if fc_layer==0:
        f.write("output = layers.Dense(10, activation = 'softmax', name='output')(output)\n\n")
    else:
        for i in range(fc_layer):
            if i==0:
                f.write("output = layers.Dense(1000, activation = actF, name='fc" + str(i) + "')(output)\n")
            else:
                f.write("output = layers.Dense(1000, activation = actF, name='fc" + str(i) + "')(dropout)\n")
            f.write("dropout = layers.Dropout(rate=drop_out)(output)\n")
        f.write("output = layers.Dense(10, activation = 'softmax', name='output')(dropout)\n\n")

    f.write("model = keras.Model(inputs = inputs, outputs = output)\n")
    f.write("model.summary()\n\n")

    f.write("model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])\n")
    f.write("hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))\n\n")

    f.write("score = model.evaluate(x_test, y_test, verbose=0)\n")
    f.write("end_time = time.time()\n")
    f.write("end_clock = time.clock()\n")
    f.write("train_time = end_time - start_time\n")
    f.write("train_clock = end_clock - start_clock\n")
    f.write("print('Time to train (time) = ', train_time)\n")
    f.write("print('Time to train (clock) = ', train_clock)\n")
    f.write("print(\"Accuracy=\", score[1], \"genetic\")\n")

    if index == None: print("wrong index")
    f.write("model.save(\'./saved/model_" + str(index) + ".h5\')\n")

    f.close()