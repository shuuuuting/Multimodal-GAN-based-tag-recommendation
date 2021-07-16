#----------------------------------------------------------------------------------------
#G input: data or feature / output: tag prob 
#D input: tag prob / output: real or false
#------------------------------------------------------------------------------------------

#%%
from keras.utils import Sequence
class DataGen(Sequence):
    def __init__(self, image_train, text_train, title_train, tag_train, tot_ex, batch_size):
	    self.batch_size = batch_size
	    self.tot_ex = tot_ex #總data長度
	    #self.on_epoch_end()
	    self.idlist = list(range(0, tot_ex))
	    self.globalidx = np.random.choice(self.idlist, int(self.tot_ex*0.5)) #epoch global training set
	    self.G_sampleprob = np.ones(self.tot_ex) / self.tot_ex
	    self.D_sampleprob = np.ones(self.tot_ex) / self.tot_ex
    
    def __getitem__(self, index): 
        if index==0: #from global set
            general_sampleidx = np.random.choice(self.globalidx, int(batch_size/2))
            (X,y) =  self.__data_generation(indices = general_sampleidx)
            return (X,y)
        elif index==1: #sampling strategy for G
            G_strategic_sampleidx = np.random.choice(self.idlist, int(batch_size/2), p = self.G_sampleprob)
            (X,y) =  self.__data_generation(indices = G_strategic_sampleidx)
            return (X,y)
        else: #sampling strategy for D
            D_strategic_sampleidx = np.random.choice(self.idlist, int(batch_size/2), p = self.D_sampleprob)
            (X,y) =  self.__data_generation(indices = D_strategic_sampleidx)
            return (X,y)

    def __len__(self):
	    return int(np.floor((self.tot_ex) / self.batch_size))

    def on_epoch_end(self): #重新抽globalset＆更新sample prob.
	    self.globalidx = np.random.choice(self.idlist, int(self.tot_ex*0.5))
	    G_out, GAN_out = gan.stacked_G_D.predict([image_train, text_train, title_train])
	    G_classloss = tf.keras.losses.binary_crossentropy(G_out, tag_train.astype(np.float32)).numpy() 
	    G_loss = G_classloss + tf.keras.losses.binary_crossentropy(GAN_out, np.ones_like(GAN_out)).numpy() 
	    if np.isnan(G_loss).any():
	        self.G_sampleprob = np.ones(self.tot_ex) / self.tot_ex
	    else:
	        self.G_sampleprob = (1/G_loss) / sum(1/G_loss)
	    if np.isnan(G_loss).any():
	        self.D_sampleprob = np.ones(self.tot_ex) / self.tot_ex
	    else:
	        self.D_sampleprob = (1/G_classloss) / sum(1/G_classloss)
	    return

    def __data_generation(self, indices):
        indices.sort()
        X = [image_train[indices], text_train[indices], title_train[indices]]
        y = tag_train[indices]
        return (X,y)

datagen = DataGen(image_train, text_train, title_train, tag_train, tot_ex=len(image_train), batch_size=64)

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
        #self.optimizer = 'adam'

        self.G = self.__generator()
        self.G.compile(loss=myLossFunc, optimizer=self.g_optimizer)

        self.D = self.__discriminator()
        self.D.compile(loss=myLossFunc, optimizer=self.d_optimizer)

        self.stacked_G_D = self.__stacked_generator_discriminator()
        self.stacked_G_D.compile(loss=[myLossFunc, myLossFunc], loss_weights = [0.6,0.4], optimizer=self.g_optimizer)


    def __generator(self):
        inputs_img = Input(shape=(7, 7, 512))
        inputs_text = Input(shape=(seq_length,))
        inputs_group = Input(shape=(10,)) #embedding
        iFeature, iFeature_pooling = imageFeature(inputs_img)
        tFeature = textFeature(inputs_text)
        gFeature = groupFeature(inputs_group) #embedding
        co_feature = coAttention_alt(dim_k=dim_k)([iFeature, tFeature, gFeature])
        dropout = Dropout(drop_rate)(co_feature)
        tag_prob = Dense(num_tags, activation="softmax", use_bias=True)(dropout)
        model = Model(inputs=[inputs_img, inputs_text, inputs_group], outputs=[tag_prob])
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

    def train(self, image_train, text_train, title_train, tag_train, epochs, batch_size):
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
                x_gen_batch = [np.append(real_x[0], tobegen_x[0],axis=0), np.append(real_x[1], tobegen_x[1],axis=0),
                                np.append(real_x[2], tobegen_x[2],axis=0)]
                y_mislabled = np.ones((batch_size, 1))
                g_loss = self.stacked_G_D.train_on_batch(x_gen_batch, [np.append(real_tags, tobegen_y, axis=0), y_mislabled])
                if cnt % 10 == 0:
                    print('iter: %d, [d_loss: %f], [g_loss: %f]' % (cnt+1, d_loss, g_loss[0]))
                    
            acc_K, _, _, _ = evaluation(tag_train, self.G.predict(x=[image_train, text_train, title_train]), TopK) 
            print('epoch: %d, [d_loss: %f], [g_loss: %f], [acc: %f]' % (e+1, d_loss, g_loss[0], acc_K)) 
            print('[class_loss: %f], [adv_loss: %f]' % (g_loss[1], g_loss[2])) 
            dlist.append(d_loss)#(g_loss[1]) 
            glist.append(g_loss[0])#g_loss[2])
            datagen.on_epoch_end()
        return dlist, glist

#%%
TopK = 10
epochs = 20
batch_size = 64
gan = GAN(num_tags)
d_history, g_history = gan.train(image_train, text_train, title_train, tag_train.astype(np.float32), epochs=epochs, batch_size=batch_size)

#%%
y_pred = gan.G.predict(x=[image_test, text_test, title_test])
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

#%% GAN/20epochs/batchsize=64
#G optimizer: adam 0.002 / D optimizer: adam 0.00002
#D 5 step per G step
#G last layer: softmax
#myLossFunc
#loss
#acc:  0.8130511463844797
#precision:  0.16437389770723107
#recall:  0.7194314968785868
#f1:  0.25279830826172167

#1/loss
#acc:  0.8156966490299824
#precision:  0.16261022927689595
#recall:  0.7157809831751629
#f1:  0.25039232086793806