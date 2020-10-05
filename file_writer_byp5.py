#!/usr/bin/env python
# coding: utf-8

# In[6]:


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
    epoch = gene[8]
    byp = gene[9]

    f = open("created_cnn.py", 'w')

    # import
    f.write("\n")
    f.write("import tensorflow as tf\n")
    f.write("from tensorflow import keras\n")
    f.write("from tensorflow.keras import layers\n")
    f.write("from tensorflow.keras import datasets\n")
    f.write("import copy\n")
    f.write("import numpy as np\n\n")

    
    #func
    f.write("def make_rand(net_list):         #파생된 레이어들을 list에 담아 반환\n")
    f.write("  lis=list()\n")
    f.write("  re_seed=random.randint(1,4)  #파생레이어 1~4개 생성\n")
    f.write("  for i in range(re_seed):\n")
    f.write("    seed=random.randint(1,4)    #한 레이어에서 파생레이어 생성\n")
    f.write("    if seed==1:\n")
    f.write("     im_output= layers.Conv2D(filters = 64, kernel_size = [ks, ks], padding = 'same', activation = actF, name='block_identity')(output)\n")
    f.write("    elif seed==2:\n")
    f.write("      im_output= layers.Dropout(rate=0.25)(output)\n")
    f.write("    elif seed==3:\n")
    f.write("     im_output= layers.MaxPooling2D(pool_size = [ks, ks], padding = 'same', strides = 1)(output)\n")
    f.write("    elif seed==4:\n")
    f.write("     im_output = layers.Activation(ActF)(output)\n")
    f.write("    lis.append(im_output)\n")
    f.write("  return lis\n")
    f.write("\n")
    
    f.write("def make_short_cut(a_layer,b_layer):  # 받은 두개의 레이어로 shortcut을 만들어 반환\n")
    f.write("  im_output = layers.Add()([a_layer,b_layer])\n")
    f.write("  return im_output\n")
    f.write("\n")
    f.write("\n")
    f.write("\n")
    
    
    
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
    f.write("epoch = " + str(epoch) + "\n")
    f.write("byp = " + str(byp) + "\n\n")

    f.write("img_rows = 28\n")
    f.write("img_cols = 28\n\n")

    f.write("(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()\n\n")

    f.write("input_shape = (img_rows, img_cols, 1)\n")
    #f.write("x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)\n")
    #f.write("x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)\n")
    #f.write("x_train = x_train.astype('float32') / 255.\n")
    #f.write("x_test = x_test.astype('float32') / 255.\n\n")

    f.write("batch_size = 128\n")
    f.write("num_classes = 10\n")
    f.write("epochs =" + str(epoch) + "\n\n")
    #f.write("epochs = 1\n\n")

    f.write("y_train = keras.utils.to_categorical(y_train, num_classes)\n")
    f.write("y_test = keras.utils.to_categorical(y_test, num_classes)\n\n")
    
    f.write("filename = 'checkpoint.h5'.format(epochs, batch_size)\n")
    f.write("early_stopping=EarlyStopping(monitor='val_loss',mode='min',patience=15,verbose=1)\n")
    f.write("checkpoint=ModelCheckpoint(filename,monitor='val_loss',verbose=1,save_best_only=True,mode='auto')\n")
    f.write("\n")

    # !
    f.write("inputs = keras.Input(shape = input_shape, name = 'input')\n")
    for i in range(depth):
        f.write("a=make_rand(net_list)\n")
        f.write("net_list.extend(a)\n")
        f.write("if len(a)==1:r_num=0 \n")
        f.write("else:r_num=random.randint(0,len(a)-1) \n")
        f.write("output=a[r_num]\n")
        f.write("\n")
        #random
        f.write("short_cut_dec=random.randint(1,5)\n")
        f.write("if short_cut_dec>=byp  and len(net_list)>1:\n")
        f.write("  add_num=add_num+1\n")
        f.write("  add_layer_num=random.randint(0,len(net_list)-1\n")
        f.write("  add_list=[] #인덱스 저장해서 같은거 있는지 확인하려고 만든 리스트\n")
        f.write("  for _ in range( random.randint(0,len(net_list)-1) ): #random개만큼 add한다\n")
        f.write("   a_layer_num=random.randint(0,len(net_list)-1)\n")
        f.write("   add_list.append(a_layer_num)\n")
        f.write("   c=make_short_cut(net_list[a_layer_num],output)\n")
        f.write("   output=c\n")
        f.write(" net_list.append(net_list)\n")
        f.write("\n")

        
        f.write("output = layers.GlobalAveragePooling2D()(inputs)\n")
            

        
        
        
        
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

    if parent != None:
        f.write("pmodel = keras.models.load_model('./saved/model_" + str(parent) + ".h5')\n")
        fp = open("./saved/chromosome_" + str(parent) + ".txt", "r")
        p_kernel_size = int(fp.readline())
        p_conv_layer = int(fp.readline())
        p_n_conv = int(fp.readline())
        p_fc_layer = int(fp.readline())
        fp.close()
        if kernel_size != p_kernel_size:
            f.write("for i in range(min(" + str(conv_layer) + "," + str(p_conv_layer) + ")):\n")
            if kernel_size < p_kernel_size:
                f.write("    try:\n")
                f.write("        k = pmodel.get_layer('block' + str(i) + '_identity').get_weights()[0]\n")
                f.write("        k = k[:" + str(kernel_size) + ", :" + str(kernel_size) + ", :, :]\n")
                f.write("        b = pmodel.get_layer('block' + str(i) + '_identity').get_weights()[1]\n")
                f.write("        w = [k,b]\n")
                f.write("        model.get_layer('block' + str(i) + '_identity').set_weights(w)\n")
                f.write("    except ValueError as e: print(e)\n")
                f.write("    for j in range(min(" + str(n_conv+1) + "," + str(p_n_conv+1) + ")):\n")
                f.write("        try:\n")
                f.write("            k = pmodel.get_layer('block' + str(i) + '_conv' + str(j)).get_weights()[0]\n")
                f.write("            k = k[:" + str(kernel_size) + ", :" + str(kernel_size) + ", :, :]\n")
                f.write("            b = pmodel.get_layer('block' + str(i) + '_conv' + str(j)).get_weights()[1]\n")
                f.write("            w = [k,b]\n")
                f.write("            model.get_layer('block' + str(i) + '_conv' + str(j)).set_weights(w)\n")
                f.write("        except ValueError as e: print(e)\n")
            else:
                padding = (kernel_size-p_kernel_size)/2
                f.write("    try:\n")
                f.write("        k = pmodel.get_layer('block' + str(i) + '_identity').get_weights()[0]\n")
                f.write("        k = np.pad(k, ((" + str(math.ceil(padding)) + "," + str(math.floor(padding)) + "),(" + str(math.ceil(padding)) + "," + str(math.floor(padding)) + "),(0,0),(0,0)), 'constant', constant_values=0)\n")
                f.write("        b = pmodel.get_layer('block' + str(i) + '_identity').get_weights()[1]\n")
                f.write("        w = [k,b]\n")
                f.write("        model.get_layer('block' + str(i) + '_identity').set_weights(w)\n")
                f.write("    except ValueError as e: print(e)\n")
                f.write("    for j in range(min(" + str(n_conv+1) + "," + str(p_n_conv+1) + ")):\n")
                f.write("        try:\n")
                f.write("            k = pmodel.get_layer('block' + str(i) + '_conv' + str(j)).get_weights()[0]\n")
                f.write("            k = np.pad(k, ((" + str(math.ceil(padding)) + "," + str(math.floor(padding)) + "),(" + str(math.ceil(padding)) + "," + str(math.floor(padding)) + "),(0,0),(0,0)), 'constant', constant_values=0)\n")
                f.write("            b = pmodel.get_layer('block' + str(i) + '_conv' + str(j)).get_weights()[1]\n")
                f.write("            w = [k,b]\n")
                f.write("            model.get_layer('block' + str(i) + '_conv' + str(j)).set_weights(w)\n")
                f.write("        except ValueError as e: print(e)\n")
        f.write("model.load_weights('./saved/model_" + str(parent) + ".h5', by_name=True, skip_mismatch=True)\n")

    f.write("hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpoint,earlystopping])\n\n")

    f.write("score = model.evaluate(x_test, y_test, verbose=0)\n")
    f.write("print(\"Accuracy=\", score[1], \"genetic\")\n")
    
    if index == None: print("wrong index")
    f.write("model.save(\'./saved/model_" + str(index) + ".h5\')\n")

    f.close()


# In[ ]:




