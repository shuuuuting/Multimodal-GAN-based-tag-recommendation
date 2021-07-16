#----------------------------------------------------------------------------------------
#G input: data or feature / output: tag prob 
#D input: tag prob / output: real or false
#------------------------------------------------------------------------------------------

#%%
from keras.utils import Sequence
import tensorflow as tf
class DataGen(Sequence):
    def __init__(self, image_train, embedding_train, group_train, title_train, tag_train, tot_ex, batch_size):
	    self.batch_size = batch_size
	    self.tot_ex = tot_ex #總data長度
	    self.lwr = 0
	    self.upr = batch_size//2
	    self.half = batch_size//2
	    self.image_train = image_train
	    self.text_train = embedding_train
	    self.group_train = group_train
	    self.title_train = title_train
	    self.idlist = list(range(0, tot_ex))
	    self.globalidx = np.random.choice(self.idlist, int(self.tot_ex*0.5)) #era global training set
	    self.G_sampleprob = np.ones(len(self.idlist)) / len(self.idlist)
	    self.D_sampleprob = np.ones(len(self.idlist)) / len(self.idlist)
	    self.G_unknownidx = np.random.choice(self.idlist, int(self.tot_ex*0.5), p = self.G_sampleprob)
	    self.D_unknownidx = np.random.choice(self.idlist, int(self.tot_ex*0.5), p = self.D_sampleprob)

    def __getitem__(self, index): 
        if index==0: #from global set
            general_sampleidx = self.globalidx[self.lwr:self.upr]
            (X,y) =  self.__data_generation(indices = general_sampleidx)
            return (X,y)
        elif index==1: #sampling strategy for G
            G_strategic_sampleidx = self.G_unknownidx[self.lwr:self.upr]
            (X,y) =  self.__data_generation(indices = G_strategic_sampleidx)
            self.lwr += self.half
            self.upr += self.half #先抽D再抽G，所以等到G再移動lwr跟upr位置
            return (X,y)
        else: #sampling strategy for D
            D_strategic_sampleidx = self.D_unknownidx[self.lwr:self.upr]
            (X,y) =  self.__data_generation(indices = D_strategic_sampleidx)
            return (X,y)

    def __len__(self):
	    return int(np.floor((self.tot_ex) / (self.batch_size)))

    def on_epoch_end(self): #更新sample prob./抽known&unknown set
	    np.random.shuffle(self.globalidx) 
	    np.random.shuffle(self.G_unknownidx) 
	    np.random.shuffle(self.D_unknownidx) 
	    self.lwr = 0
	    self.upr = self.batch_size//2
	    return
    
    def on_era_end(self): #重新抽globalset
	    G_out, GAN_out = gan.stacked_G_D.predict([self.image_train, self.text_train, self.group_train, self.title_train])
	    G_classloss = tf.keras.losses.binary_crossentropy(G_out, tag_train.astype(np.float32)).numpy() 
	    G_loss = G_classloss + tf.keras.losses.binary_crossentropy(GAN_out, np.ones_like(GAN_out)).numpy() 
	    if np.isnan(G_loss).any():
	        self.G_sampleprob = np.ones(len(self.idlist)) / len(self.idlist)
	    else:
	        self.G_sampleprob = (G_loss) / sum(G_loss)
	    if np.isnan(G_loss).any():
	        self.D_sampleprob = np.ones(len(self.idlist)) / len(self.idlist)
	    else:
	        self.D_sampleprob = (G_classloss) / sum(G_classloss)
	    self.globalidx = np.random.choice(self.idlist, int(self.tot_ex*0.5)) #抽50%作為era global set
	    self.G_unknownidx = np.random.choice(self.idlist, int(self.tot_ex*0.5), p = self.G_sampleprob)
	    self.D_unknownidx = np.random.choice(self.idlist, int(self.tot_ex*0.5), p = self.D_sampleprob)
	    return

    def __data_generation(self, indices):
        indices.sort()
        X = [self.image_train[indices], self.text_train[indices], self.group_train[indices], self.title_train[indices]]
        y = tag_train[indices]
        return (X,y)

datagen = DataGen(image_train, embedding_train, group_train, title_train, tag_train, tot_ex=len(image_train), batch_size=64)

#%%
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.models import Sequential
import random 
from numpy.random import RandomState
class GAN(object):
    def __init__(self, num_tags):
        self.num_tags = num_tags
        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
        self.d_optimizer = Adam(lr=0.00002, beta_1=0.5, decay=8e-8)
        self.g_optimizer = Adam(lr=0.002, beta_1=0.5, decay=8e-8)
        #self.g_optimizer = 'adam'

        self.G = self.__generator()
        self.G.compile(loss=myLossFunc, optimizer=self.g_optimizer)

        self.D = self.__discriminator()
        self.D.compile(loss=myBinaryLossFunc, optimizer=self.d_optimizer)

        self.stacked_G_D = self.__stacked_generator_discriminator()
        self.stacked_G_D.compile(loss=[myLossFunc, myBinaryLossFunc], loss_weights = [0.6,0.4], optimizer=self.g_optimizer)


    def __generator(self):
        inputs_img = Input(shape=(7, 7, 512))
        inputs_text = Input(shape=(seq_length,200))
        inputs_group = Input(shape=(embedding_size,)) #doc2vec
        inputs_channel = Input(shape=(seq_length,))
        inputs_title = Input(shape=(10,)) #embedding
        '''
        iCNN = load_model('pretrained_iCNN.h5')
        tCNN = load_model('pretrained_tCNN.h5')
        iCNN = Model(inputs=iCNN.input, outputs=iCNN.get_layer('icnn').output)
        tCNN = Model(inputs=tCNN.input, outputs=tCNN.get_layer('tcnn').output)
        iFeature = iCNN(inputs_img)
        tFeature = tCNN(inputs_text)
        
        iCNN = load_model('pretrained_iCNN.h5', custom_objects={'SeqSelfAttention': SeqSelfAttention})
        tCNN = load_model('pretrained_tCNN.h5', custom_objects={'SeqSelfAttention': SeqSelfAttention})
        iCNN = Model(inputs=iCNN.input, outputs=iCNN.get_layer('icnn').output)
        tCNN = Model(inputs=tCNN.input, outputs=tCNN.get_layer('tcnn').output)
        iFeature = iCNN(inputs_img)
        tFeature = tCNN(inputs_text)
        iFeature = SeqSelfAttention(attention_activation='sigmoid')(iFeature)
        tFeature = SeqSelfAttention(attention_activation='sigmoid')(tFeature)
        iFeature.set_shape((inputs_img.shape[0],49,200))
        tFeature.set_shape((inputs_text.shape[0],50,200))
        '''
        iFeature, _ = imageFeature(inputs_img)
        tFeature = Conv1D(200, 3, padding='same', strides=1, activation='relu')(inputs_text)
        gFeature = inputs_group #doc2vec
        #gFeature = groupFeature(inputs_channel)
        eFeature = groupFeature(inputs_title) #embedding
        co_feature = coAttention_alt(dim_k=dim_k)([iFeature, tFeature, gFeature, eFeature])
        dropout = Dropout(drop_rate)(co_feature)
        tag_prob = Dense(num_tags, activation="softmax", use_bias=True)(dropout)
        model = Model(inputs=[inputs_img, inputs_text, inputs_group, inputs_title], outputs=[tag_prob])
        #model.summary()
        return model 

    def __discriminator(self):
        disc_in = Input(shape=(num_tags,))
        x = Dense(num_tags, activation = LeakyReLU(alpha=.2))(disc_in) 
        x = Dense(256, activation = LeakyReLU(alpha=.2))(x)
        x = Dense(64, activation = LeakyReLU(alpha=.2))(x)
        disc_out = Dense(1, activation = 'sigmoid', name = "Discriminator")(x)
        model = Model(inputs=[disc_in], outputs=[disc_out]) 
        #model.summary()
        return model

    def __stacked_generator_discriminator(self):
        self.D.trainable = False
        gan_input = self.G.input
        gen_output = self.G(gan_input)
        gan_output = self.D(gen_output)
        model = Model(inputs=gan_input, outputs=[gen_output, gan_output])
        model.summary()
        return model

    def train(self, image_train, embedding_train, group_train, title_train, tag_train, epochs, batch_size):
        dlist, glist = [], []
        global datagen
        iters = datagen.__len__()
        for e in range(epochs):
            for cnt in range(iters):
                ## train discriminator ##  
                #fetch a batch of data to train on 
                (real_x, real_tags) = datagen.__getitem__(index=0)
                #now that we have our real example we must use self.G.predict to get our 'fake' examples	
                (tobefake_x, _) = datagen.__getitem__(index=2)
                fake_tags = self.G.predict(x=tobefake_x)
                #adjust real_tags from 0/1 to (0-0.2)/(0.4-0.6)
                #real_tags = (real_tags*0.4) + np.random.uniform(0,0.2,size=real_tags.shape)
                #real_tags = (real_tags*0.4) + np.random.random(real_tags.shape)*0.2
                real_tags = (real_tags*0.7) + np.random.random(real_tags.shape)*0.3

				#now that we have our real and fake labels we can train discriminator 
                x_combined_batch = np.append(real_tags, fake_tags, axis=0)
                y_combined_batch = np.concatenate((np.ones((int(batch_size/2), 1)), np.zeros((int(batch_size/2), 1)))) 
                    
                #把真跟假打散
                seed = random.randint(0,10000)
                p = RandomState(seed)
                p.shuffle(x_combined_batch)
                p = RandomState(seed)
                p.shuffle(y_combined_batch)
                self.D.trainable = True
                if cnt % 5 == 0:
                    d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)

                ## train generator ## 
                (tobegen_x, tobegen_y) = datagen.__getitem__(index=1)
                x_gen_batch = [np.append(real_x[0], tobegen_x[0],axis=0), np.append(real_x[1], tobegen_x[1],axis=0)
                                ,np.append(real_x[2], tobegen_x[2],axis=0), np.append(real_x[3], tobegen_x[3],axis=0)]
                y_mislabled = np.ones((batch_size, 1))
                g_loss = self.stacked_G_D.train_on_batch(x_gen_batch, [np.append(real_tags, tobegen_y, axis=0), y_mislabled])
                if cnt % 10 == 0:
                    print('iter: %d, [d_loss: %f], [g_loss: %f]' % (cnt+1, d_loss, g_loss[0]))
                    
            acc_K, _, _, _ = evaluation(tag_train, self.G.predict(x=[image_train, embedding_train, group_train, title_train]), TopK) 
            print('epoch: %d, [d_loss: %f], [g_loss: %f], [acc: %f]' % (e+1, d_loss, g_loss[0], acc_K)) 
            print('[class_loss: %f], [adv_loss: %f]' % (g_loss[1], g_loss[2])) 
            dlist.append(d_loss)#(g_loss[1]) 
            glist.append(g_loss[0])#g_loss[2])
            datagen.on_epoch_end()
            if e % 5 == 0: #5 epochs as 1 era
                datagen.on_era_end()
        return dlist, glist

#%%
TopK = 10
epochs = 20
batch_size = 64
gan = GAN(num_tags)
d_history, g_history = gan.train(image_train, embedding_train, group_train, title_train, tag_train.astype(np.float32), epochs=epochs, batch_size=batch_size)

#%%
y_pred = gan.G.predict(x=[image_test, embedding_test, group_test, title_test])
acc_K, precision_K, recall_K, f1_K = evaluation(tag_test, y_pred, TopK)
print('acc: ', acc_K)
print('precision: ', precision_K)
print('recall: ', recall_K)
print('f1: ', f1_K)

import matplotlib.pyplot as plt
plt.title('GAN loss')
plt.xlabel('epoch')
plt.plot(range(epochs), d_history, label='D')
plt.plot(range(epochs), g_history, label='G')
plt.legend()
plt.show()

#%%
gan_out_test = gan.D.predict(y_pred)
mislabled_test = np.ones((len(tag_test), 1))
loss_test = myLossFunc(mislabled_test, gan_out_test) 

#%%GAN sampling strategy 10
#era(5 epochs): global 0.5 & unknown 0.5
#epoch: knownset = global / unknownset resample 
#real sample from knownset
#to_be_fake sample from D_unkownset
#to_be_gen sample from G_unkownset
#D input: real batchsize/2 + G(to_be_fake) batchsize/2
#G input: real batchsize/2 + to_be_gen batchsize/2
#20epochs/batchsize=64
#G optimizer: adam 0.002 / D optimizer: adam 0.00002
#D 5 step per G step
#G last layer: softmax
#(pretrain) myLosss+myBinaryLoss
#acc:  0.7283950617283951
#precision:  0.140652557319224
#recall:  0.6000934324347021
#f1:  0.21545966932206226

#(no pretrain) myLosss+myBinaryLoss
#acc:  0.763668430335097
#precision:  0.14876543209876542
#recall:  0.6457535483329133
#f1:  0.2283766879813778




