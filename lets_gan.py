from __future__ import print_function

import tensorflow as tf

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
# from keras.optimizers import *
from keras.utils import np_utils
import keras
import keras.backend as K

import math
import random

import sgdr
import numpy as np

import cv2

batch_size = 32
nb_classes = 10
nb_epoch = 200
eps=1e-11

zed = 32

# data_augmentation = True

def cifar():
    # input image dimensions
    img_rows, img_cols = 32, 32
    # the CIFAR10 images are RGB
    img_channels = 3

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # X_train = np.vstack([X_train,X_test[0:5000]])
    # y_train = np.vstack([y_train,y_test[0:5000]])
    #
    # X_test = X_test[5000:10000]
    # y_test = y_test[5000:10000]

    # X_train = X_train[:,8:24,8:24,:]

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    X_train-=0.5
    X_test-=0.5

    return X_train,Y_train,X_test,Y_test
print('loading cifar...')
xt,yt,xv,yv = cifar()

from keras_resnet import relu,cake,gen_cake
def relu(i):
    return LeakyReLU(.2)(i)

def bn(i):
    return BatchNormalization()(i)

def gen(): # generative network
    inp = Input(shape=(128,))
    # inp = Input(shape=(32,32,3))
    i = inp

    # i = Reshape((4,4,8))(i)
    # # i = UpSampling2D(size=(8,8))(i)

    i = Reshape((1,1,128))(i)

    i = gen_cake(128,64,layers=3,std=2,inputh=1,inputw=1,bsize=batch_size)(i)
    i = gen_cake(64,48,layers=3,std=2,inputh=2,inputw=2,bsize=batch_size)(i)
    i = gen_cake(48,32,layers=3,std=2,inputh=4,inputw=4,bsize=batch_size)(i)

    i = gen_cake(32,24,layers=3,std=2,inputh=8,inputw=8,bsize=batch_size)(i)
    i = gen_cake(24,24,layers=3,std=2,inputh=16,inputw=16,bsize=batch_size)(i)
    i = gen_cake(24,16,layers=3,std=1,inputh=32,inputw=32,bsize=batch_size)(i)

    i = relu(i)
    # i = Deconvolution2D(3,3,3,border_mode='same',output_shape=(batch_size,32,32,3))(i)

    i = Convolution2D(3,1,1,border_mode='same')(i)

    i = Activation('tanh')(i)

    m = Model(input=inp,output=i)
    mf = Model(input=inp,output=i)
    mf.trainable = False
    return m,mf

def gen2(): # generative network, 2
    inp = Input(shape=(zed,))
    # inp = Input(shape=(32,32,3))
    i = inp
    i = Reshape((1,1,zed))(i)

    ngf=24

    def deconv(i,nop,kw,oh,ow,std=1,tail=True,bm='same'):
        global batch_size
        i = Deconvolution2D(nop,kw,kw,subsample=(std,std),border_mode=bm,output_shape=(batch_size,oh,ow,nop))(i)
        if tail:
            i = bn(i)
            i = relu(i)
        return i

    i = deconv(i,nop=ngf*8,kw=4,oh=4,ow=4,std=1,bm='valid')
    i = deconv(i,nop=ngf*4,kw=4,oh=8,ow=8,std=2)
    i = deconv(i,nop=ngf*2,kw=4,oh=16,ow=16,std=2)
    i = deconv(i,nop=ngf*1,kw=4,oh=32,ow=32,std=2)

    i = deconv(i,nop=3,kw=2,oh=32,ow=32,std=1,tail=False) # out : 32x32
    i = Activation('tanh')(i)

    m = Model(input=inp,output=i)
    mf = Model(input=inp,output=i)
    mf.trainable = False
    return m,mf

def concat_diff(i):
    # return i
    bv = Lambda(lambda x:K.mean(K.abs(x[:] - K.mean(x,axis=0)),axis=-1,keepdims=True))(i)
    i = merge([i,bv],mode='concat')
    return i

def dis(): # discriminative network
    # inp = Input(shape=(None,None,3))
    inp = Input(shape=(32,32,3))
    i = inp

    i = Convolution2D(16,3,3,border_mode='same')(i)

    i = cake(16,16,3,2)(i)#16
    i = cake(16,24,3,2)(i)#

    i = concat_diff(i)

    i = cake(48,48,3,2)(i)#4
    i = cake(48,64,3,2)(i)#
    i = cake(64,128,3,2)(i)#1,1,128

    i = concat_diff(i)

    i = Activation('linear',name='conv_exit')(i)
    i = relu(i)

    i = Reshape((256,))(i)


    # mbatch_var = Lambda(lambda x:x[1][:,0:1]*0.0 - K.mean(K.var(x[0],axis=0)))([inp,i])
    # i = merge([i,mbatch_var],mode='concat')


    # i = Convolution2D(1,1,1)(i) #1,1,1
    # i = Reshape((1,))(i)
    i=Dense(1)(i)

    i = Activation('sigmoid')(i) # output 1 for real, 0 for fake

    m = Model(input=inp,output=i)
    mf = Model(input=inp,output=i)
    mf.trainable = False
    return m,mf

def dis2(): # discriminative network, 2
    # inp = Input(shape=(None,None,3))
    inp = Input(shape=(32,32,3))
    i = inp

    ndf=24

    # i = Convolution2D(ndf,4,4,border_mode='same',subsample=(1,1))(i)
    # i = relu(i)

    def conv(i,nop,kw,std=1,usebn=True,bm='same'):
        i = Convolution2D(nop,kw,kw,border_mode=bm,subsample=(std,std))(i)
        if usebn:
            i = bn(i)
        i = relu(i)
        return i

    i = conv(i,ndf*1,4,std=2,usebn=False)
    i = concat_diff(i)
    i = conv(i,ndf*2,4,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*4,4,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*8,4,std=2)
    i = concat_diff(i)

    # in: 2x2
    # i = conv(i,n,2,std=1)
    # 1x1
    i = Convolution2D(1,2,2,border_mode='valid')(i)

    # i = concat_diff(i)

    i = Activation('linear',name='conv_exit')(i)
    i = Activation('sigmoid')(i)

    i = Reshape((1,))(i)

    m = Model(input=inp,output=i)
    mf = Model(input=inp,output=i)
    mf.trainable = False
    return m,mf


print('generating G...')
gm,gmf = gen2()
gm.summary()

print('generating D...')
dm,dmf = dis2()
dm.summary()

def gan(gm,dm):
    # this is the fastest way to train a GAN in Keras
    # two models are updated simutaneously in one pass

    noise = Input(shape=(zed,))
    # noise2 = Input(shape=(zed,))
    real_image = Input(shape=(32,32,3))

    generated = gm(noise)
    gscore = dm(generated)
    rscore = dm(real_image)

    # def ccel(x):
    #     gs=x[0]
    #     rs=x[1]
    #     loss = - (K.log(1-gs+eps) + 0.1 * K.log(1-rs+eps) + 0.9 * K.log(rs+eps)) #sside lbl smoothing
    #     return loss
    #
    # def calc_output_shape(input_shapes):
    #     return input_shapes[0]

    # dloss = merge([gscore,rscore],mode=ccel,output_shape=calc_output_shape,name='dloss')
    # out = Lambda(lambda x:- K.log(1-x[0]) - K.log(x[1]))([gscore,rscore])

    # dm_trainer = Model(input=[real_image,noise],output=[gscore,rscore])

    def log_eps(i):
        return K.log(i+1e-11)

    # single side label smoothing: replace 1.0 with 0.9
    dloss = - K.mean(log_eps(1-gscore) + .1 * log_eps(1-rscore) + .9 * log_eps(rscore))
    gloss = - K.mean(log_eps(gscore))

    Adam = tf.train.AdamOptimizer

    lr,b1 = 1e-4,.2
    optimizer = Adam(lr,beta1=b1)

    grad_loss_wd = optimizer.compute_gradients(dloss, dm.trainable_weights)
    update_wd = optimizer.apply_gradients(grad_loss_wd)

    # optimizer = Adam(1e-4)

    grad_loss_wg = optimizer.compute_gradients(gloss, gm.trainable_weights)
    update_wg = optimizer.apply_gradients(grad_loss_wg)

    def get_internal_updates(model):
        # get all internal update ops (like moving averages) of a model
        inbound_nodes = model.inbound_nodes
        input_tensors = []
        for ibn in inbound_nodes:
            input_tensors+= ibn.input_tensors
        updates = [model.get_updates_for(i) for i in input_tensors]
        return updates

    other_parameter_updates = [get_internal_updates(m) for m in [dm,gm]]
    # those updates includes batch norm.

    print('other_parameter_updates for the models(mainly for batch norm):')
    print(other_parameter_updates)

    train_step = [update_wd, update_wg, other_parameter_updates]
    losses = [dloss,gloss]

    learning_phase = tf.get_default_graph().get_tensor_by_name(
        'keras_learning_phase:0')

    def gan_feed(sess,batch_image,z_input):
        nonlocal train_step,losses,noise,real_image,learning_phase

        res = sess.run([train_step,losses],feed_dict={
        noise:z_input,
        real_image:batch_image,
        learning_phase:True,
        # Keras layers needs to know whether
        # this run is training or testring (you know, batch norm and dropout)
        })

        loss_values = res[1]
        return loss_values #[dloss,gloss]

    # K.get_session().run(tf.global_variables_initializer())

    return gan_feed

    # def thru(y_true,y_pred):
    #     return y_pred
    #
    # g2 = gm(noise2)
    # gscore2 = dmf(g2)
    # gloss = Lambda(lambda x:- K.log(x+eps),name='gloss')(gscore2)
    #
    # gm_trainer = Model(input=noise2,output=gloss)
    #
    # lr,b1 = 1e-4,.2
    #
    # gan_trainer = Model(input=[real_image,noise,noise2],output=[dloss,gloss])
    # gan_trainer.compile(loss=[thru,thru],optimizer=Adam(lr=lr,beta_1=b1))

    # dm_trainer.compile(loss=thru,optimizer=Adam(lr=lr,beta_1=b1))
    # gm_trainer.compile(loss=thru,optimizer=Adam(lr=lr,beta_1=b1))

    # return gan_trainer
    # return dm_trainer,gm_trainer,gan_trainer

print('generating GAN...')
# dmt,gmt,gant = gan(gm,gmf,dm,dmf)
# gant = gan(gm,gmf,dm,dmf)
gan_feed = gan(gm,dm)

print('Ready. enter r() to train')

def r(ep=1000,noise_level=.01):
    sess = K.get_session()

    for i in range(ep):
        noise_level *= 0.99
        print('---------------------------')
        print('iter',i,'noise',noise_level)
        batches = 1
        total = batch_size*batches
        indices = np.random.choice(50000,total,replace=False)

        subset_cifar = np.take(xt,indices,axis=0)
        subset_cifar += np.random.normal(loc=0.,scale=noise_level,size=subset_cifar.shape) # add gaussian term

        z_input = np.random.normal(loc=0.,scale=1.,size=(total,zed))
        # z_input2 = np.random.normal(loc=0.,scale=1.,size=(total,zed))
        # dummy = np.zeros((total,1))

        # dmt.fit([subset_cifar,z_input],
        # dummy,
        # batch_size=batch_size,
        # nb_epoch=1
        # )
        #
        # gmt.fit(z_input2,
        # dummy,
        # batch_size=batch_size,
        # nb_epoch=1
        # )

        losses = gan_feed(sess,subset_cifar,z_input)
        # print(losses)
        print('dloss:{:6.4f} gloss:{:6.4f}'.format(losses[0],losses[1]))

        # gant.fit([subset_cifar,z_input,z_input2],
        # [dummy,dummy],
        # batch_size=batch_size,
        # nb_epoch=1
        # )

        if i==ep-1 or i % 10==0: show()

def autoscaler(img):
    limit = 400.
    # scales = [0.1,0.125,1./6.,0.2,0.25,1./3.,1./2.] + range(100)
    scales = np.hstack([1./np.linspace(10,2,num=9), np.linspace(1,100,num=100)])

    imgscale = limit/float(img.shape[0])
    for s in scales:
        if s>=imgscale:
            imgscale=s
            break

    img = cv2.resize(img,dsize=(int(img.shape[1]*imgscale),int(img.shape[0]*imgscale)),interpolation=cv2.INTER_NEAREST)

    return img,imgscale

def flatten_multiple_image_into_image(arr):
    import cv2
    num,uh,uw,depth = arr.shape

    patches = int(num+1)
    height = int(math.sqrt(patches)*0.9)
    width = int(patches/height+1)

    img = np.zeros((height*uh+height, width*uw+width, 3),dtype='float32')

    index = 0
    for row in range(height):
        for col in range(width):
            if index>=num-1:
                break
            channels = arr[index]
            img[row*uh+row:row*uh+uh+row,col*uw+col:col*uw+uw+col,:] = channels
            index+=1

    img,imgscale = autoscaler(img)

    return img,imgscale

def show(save=False):
    i = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))
    gened = gm.predict([i])

    gened *= 0.5
    gened +=0.5

    im,ims = flatten_multiple_image_into_image(gened)
    cv2.imshow('gened scale:'+str(ims),im)
    cv2.waitKey(1)

    if save!=False:
        cv2.imwrite(save,im*255)
