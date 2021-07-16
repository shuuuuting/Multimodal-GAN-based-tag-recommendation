#%%
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from keras.layers import Input, Dense, Embedding, Dropout, Lambda, BatchNormalization

class Generator(Model):
    def __init__(self, num_tags): 
        super(Generator, self).__init__()
        #define network layer
        self.dropout = Dropout(drop_rate)
        self.sigmoid = Dense(num_tags, activation="sigmoid", use_bias=True)
        
        #self.dense128_numeric = keras.layers.Dense( 128 ,kernel_regularizer=l2(0.05))
        #self.dense64_numeric = keras.layers.Dense( 64 ,kernel_regularizer=l2(0.05))
        #self.dense32_numeric = keras.layers.Dense( 32 ,kernel_regularizer=l2(0.05)) 
    
    def feature_extract(self, inputs_img, inputs_text) :
        iFeature, iFeature_pooling = imageFeature(inputs_img)
        tFeature = textFeature(inputs_text)
        co_feature = coAttention_alt(dim_k=dim_k)([iFeature, tFeature])
        dropout = self.dropout(co_feature)
        return dropout   

    def predict_prob(self, inputs_img, inputs_text):
        feature = self.feature_extract(inputs_img, inputs_text)
        tagprob = self.sigmoid(feature)
        #x_numeric = keras.layers.BatchNormalization()(x_numeric)
        #x_numeric = self.dense128_numeric(x_numeric)    
        #x_numeric = self.dense64_numeric(x_numeric)    
        #x_numeric = self.dense32_numeric(x_numeric)     
        return tagprob
    
    def call(self, inputs):
        inputs_img = inputs[0]
        inputs_text = inputs[1]
        tagprob = self.predict_prob(inputs_img, inputs_text)
        return tagprob

class Discriminator(Model):
    def __init__(self, num_tags):        
        super(Discriminator, self).__init__()
        #define network layer
        self.dense1 = Dense(num_tags, activation = LeakyReLU(alpha=.2))
        self.dense2 = Dense(256, activation = LeakyReLU(alpha=.2)) 
        self.dense3 = Dense(64, activation = LeakyReLU(alpha=.2))    
        self.sigmoid = Dense(1, activation = 'sigmoid')  
        #self.dense4 = Dense(2, activation = LeakyReLU(alpha=.2))
            
    def call(self, prob):
        x = self.dense1(prob)
        x = self.dense2(x)
        x = self.dense3(x)
        #realorfake = self.dense4(x)
        realorfake = self.sigmoid(x)
        return realorfake

#%%
def model_initial(G_learning_rate, D_learning_rate):
    global G, D, G_optimizer, D_optimizer, G_train_loss, D_train_loss
    G = Generator(num_tags)
    D = Discriminator(num_tags)
    
    G_optimizer = tf.keras.optimizers.Adam(learning_rate = G_learning_rate) 
    D_optimizer = tf.keras.optimizers.Adam(learning_rate = D_learning_rate)
    
    G_train_loss = tf.keras.metrics.Mean(name='G_train_loss')
    D_train_loss = tf.keras.metrics.Mean(name='D_train_loss')
    
    print("model_initial_complete!")

#%%
def generator_loss(gen_out, y_true):
    #classification_loss = keras.losses.binary_crossentropy(gen_out, y_true.astype(np.float32))
    classification_loss = tf.losses.BinaryCrossentropy(from_logits=True)(gen_out, y_true.astype(np.float32))
    #classification_loss = tf.reduce_mean(tf.nn.l2_loss(gen_out - y_true.astype(np.float32)))

    adv_loss = tf.abs(tf.math.log(1 - D(gen_out)))
    adv_loss = tf.reshape(adv_loss, [-1])
    #adv_loss = tf.reduce_sum(tf.math.log(y_true.astype(np.float32))) - tf.reduce_sum(tf.math.log(1 - D(gen_out)))
    #adv_loss = tf.reduce_mean(adv_loss)
    
    total_loss = adv_loss + classification_loss 
    
    return total_loss 

def discriminator_loss(disc_out, disc_label):    
    #disc_out包含真假資料經過discriminator辨別出來的結果
    #disc_loss = tf.reduce_mean(tf.nn.l2_loss(disc_out - disc_label))
    #disc_loss = keras.losses.binary_crossentropy(disc_out, disc_label)
    disc_loss = tf.losses.BinaryCrossentropy(from_logits=True)(disc_out, disc_label)
    #disc_loss = tf.nn.sigmoid_cross_entropy_with_logits(disc_out, disc_label)  
    return disc_loss

#%%
def G_train_step(gen_in, y_true):
    with tf.GradientTape() as G_tape :
        G_loss = generator_loss(G(gen_in), y_true)

    # 防止梯度爆炸
    if True in tf.math.is_nan(G_loss) or tf.math.count_nonzero(G_loss) < G_loss.shape[0]:   
        return

    gradients = G_tape.gradient(G_loss, G.trainable_variables)
    G_optimizer.apply_gradients(zip(gradients, G.trainable_variables))
    G_train_loss(G_loss)
      
def D_train_step(disc_in, disc_label):
    with tf.GradientTape() as D_tape :    
        disc_out = D(disc_in)
        D_loss = discriminator_loss(disc_out, disc_label)
        
    gradients = D_tape.gradient(D_loss, D.trainable_variables)
    D_optimizer.apply_gradients(zip(gradients, D.trainable_variables))
    D_train_loss(D_loss)

#%%
def x_y_concatenate(x, y, label):
    num_of_data = x[0].shape[0]
    feature = y
    
    if label == 0 :
        label = tf.zeros([num_of_data, 1])
    elif label == 1 :
        label = tf.ones([num_of_data, 1])
    return feature, label


def update_sample_prob(X_train, Y_train):
    classification_sample_weight = keras.losses.binary_crossentropy(Y_train, G(X_train)).numpy() 
    reward_sample_weight = generator_loss(G(X_train), Y_train)
    
    if np.isnan(classification_sample_weight).any() or np.any(0):
        classification_sample_weight = np.ones(len(classification_sample_weight)) / len(classification_sample_weight)
        
    if np.isnan(reward_sample_weight).any() or np.any(0):
        reward_sample_weight = np.ones(len(reward_sample_weight)) / len(reward_sample_weight)        

    #對預測準的資料加強訓練
    #epsilon = 0.00005
    #classification_sample_weight = 1 / (classification_sample_weight + epsilon)
    #reward_sample_weight = 1 / (reward_sample_weight + epsilon)

    #正規化
    classification_sample_weight = preprocessing.normalize([classification_sample_weight]).ravel()
    classification_sample_prob = classification_sample_weight / sum(classification_sample_weight)

    reward_sample_weight = preprocessing.normalize([reward_sample_weight]).ravel()
    reward_sample_prob = reward_sample_weight / sum(reward_sample_weight)

    return classification_sample_prob, reward_sample_prob

#%%
def training_evaluate(history, epoch, X_train, Y_train, X_test, Y_test):

    template = '\n ***** Epoch {}, G Loss: {} , D_loss {} ***** \n'
    print(template.format(epoch, history['G_loss'][-1], history['D_loss'][-1]))
    
    real_train = Y_train
    predict_train = G(X_train)
    real_test = Y_test
    predict_test = G(X_test)
    
    train_acc, _, _, _ = evaluation(real_train, np.array(predict_train), TopK)
    test_acc, _, _, _ = evaluation(real_test, np.array(predict_test), TopK)
    
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)  
    
    print('\n ===== training data ===== ')
    print( 'Accuracy:', train_acc)  
    
    print('\n ===== testing data ===== ')
    print( 'Accuracy:', test_acc)   
    
    print('\nbest train Accuracy:', np.max(history['train_acc']) )            
    print('best test Accuracy:', np.max(history['test_acc'])) 

#%%
def GAN_training(X_train, Y_train ,X_test ,Y_test):
    onehot_encoder = OneHotEncoder(sparse=False)
    history = {'G_loss':[], 'D_loss':[], 'train_acc':[], 'test_acc':[]}
    
    train_ids = np.arange(len(X_train[0]))
    train_ids_prob = np.ones(len(X_train[0])) / len(X_train[0]) #initial是全部樣本一樣機率
      
    for epoch in range(1, epochs+1):
        #update the prob. of samples
        classification_sample_prob, reward_sample_prob = update_sample_prob(X_train, Y_train)
        global_dataset_ids = np.random.choice(train_ids, 1024)
        
        #D training step
        for i in range(10):
            '''
            # --- using sampling strategy ---
            D_knownSet_sample_idx = np.random.choice(global_dataset_ids, int(batch_size*(1/2)))
            D_unknownSet_sample_idx = np.random.choice(train_ids ,int(batch_size*(1/2)), p = classification_sample_prob) # mse越小越會抽到

            x_real = [X_train[0][D_knownSet_sample_idx], X_train[1][D_knownSet_sample_idx]] 
            y_real = Y_train[D_knownSet_sample_idx]
            
            x_fake = [X_train[0][D_unknownSet_sample_idx], X_train[1][D_unknownSet_sample_idx]]
            y_fake = G(x_fake)            
            #print("real and fake:",x_real.shape,ys_real.shape,x_fake.shape,ys_fake.shape)
            '''
            # --- just random sampling ---
            D_knownSet_sample_idx = np.random.choice(global_dataset_ids, int(batch_size*(1/2)))
            D_unknownSet_sample_idx = np.random.choice(train_ids ,int(batch_size*(1/2)))

            x_real = [X_train[0][D_knownSet_sample_idx], X_train[1][D_knownSet_sample_idx]] 
            y_real = Y_train[D_knownSet_sample_idx]
            
            x_fake = [X_train[0][D_unknownSet_sample_idx], X_train[1][D_unknownSet_sample_idx]]
            y_fake = G(x_fake)

            #產生overview的Real與fake
            real_input, real_label = x_y_concatenate(x_real, y_real, label = 1)
            real_label = (real_label*0.7) + np.random.random(real_label.shape)*0.3        
            fake_input, fake_label = x_y_concatenate(x_fake, y_fake, label = 0)
            #print("real and fake input label:",real_input.shape,real_label.shape,fake_input.shape,fake_label.shape)
            D_input = tf.concat([real_input, fake_input], 0)
            D_label = tf.concat([real_label, fake_label], 0)
            #print("d_train_step",D_input.shape,D_label.shape)
            D_train_step(D_input , D_label)
            
        #G training step
        for i in range(50):
            '''
            # --- using sampling strategy ---
            G_knownSet_sample_idx = np.random.choice(global_dataset_ids, int(batch_size*(1/2))) 
            G_unknownSet_sample_idx = np.random.choice(train_ids, int(batch_size*(1/2)), p = reward_sample_prob) # reward_loss越小越容易抽到
            
            G_sample_indices = np.append(G_knownSet_sample_idx, G_unknownSet_sample_idx)
            '''
            # --- just random sampling ---
            G_sample_indices = np.random.choice(global_dataset_ids, int(batch_size))
            
            x = [X_train[0][G_sample_indices], X_train[1][G_sample_indices]]
            y = Y_train[G_sample_indices]
            
            G_train_step(x, y)    
            
        # Reset the metrics for the next epoch
        G_loss = G_train_loss.result()
        D_loss = D_train_loss.result()
            
        history['G_loss'].append(G_loss)
        history['D_loss'].append(D_loss)
            
        G_train_loss.reset_states()
        D_train_loss.reset_states()
    
        #if (epoch) % 5 == 0 :
        training_evaluate(history, epoch, X_train, Y_train, X_test, Y_test)                  
    return history 

def save_experiment_result(training_history):     
    plt.title('Loss')
    plt.plot(training_history['G_loss'], label='G')
    plt.plot(training_history['D_loss'], label='D')      
    plt.legend() 
    plt.show()
    
    plt.title('Accuracy')
    plt.plot(training_history['train_acc'], label='train')
    plt.plot(training_history['test_acc'], label='test')  
    plt.legend()    
    plt.show()

#%%
#超參數設定
epochs = 20
batch_size = 64

G_learning_rate = 0.002
D_learning_rate = 0.00002

def main():
    for i in range(1):
        model_initial(G_learning_rate, D_learning_rate)
        training_history = GAN_training([image_train, text_train], tag_train, [image_test, text_test], tag_test)
        save_experiment_result(training_history)

if __name__ == '__main__':     
    main()  
# %%
