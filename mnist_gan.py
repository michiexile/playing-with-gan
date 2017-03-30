#import tensorflow
import keras
import keras.datasets.mnist
from keras.layers import *
from keras.models import *

from scipy import *
import numpy.random
import random

# Fix issue with Dropout layers
#tensorflow.python.control_flow_ops = tensorflow

keras.backend.set_image_dim_ordering('th')

# Pick optimizer
opt = keras.optimizers.Adam(lr=1e-4)
dopt = keras.optimizers.Adam(lr=1e-3)

# Load MNIST
import gzip
from six.moves import cPickle
f = gzip.open('/scratch/m.johansson/playing-with-gan/mnist.pkl.gz', 'rb')
(X_train, y_train), (X_test, y_test) = cPickle.load(f)
_,img_row,img_col = X_train.shape
X_train = X_train.astype('float32') / 255
X_train = X_train.reshape(X_train.shape[0], 1, img_row, img_col)
X_test = X_test.astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 1, img_row, img_col)
dropout_rate = 0.25

shp = X_train.shape

# Generator
nch = 200
g_input = Input(shape=[100])
H = Dense(nch*14*14, init='glorot_normal')(g_input)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Reshape([nch,14,14])(H)
H = UpSampling2D(size=(2,2))(H)
H = Convolution2D(nch//2,3,3,border_mode='same', init='glorot_normal')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(nch//4,3,3,border_mode='same', init='glorot_normal')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(1,1,1,border_mode='same', init='glorot_normal')(H)
g_V = Activation('sigmoid')(H)
generator = Model(g_input, g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)

# Discriminator
d_input = Input(shape=shp)
H = Convolution2D(256, 5, 5, subsample=(2,2), border_mode='same', activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(256, 5, 5, subsample=(2,2), border_mode='same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2, activation='softmax')(H)
discriminator = Model(d_input, d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)



# Start / stop training
def make_trainable(network, value):
    network.trainable = value
    for layer in network.layers:
        layer.trainable = value

make_trainable(discriminator, False)



# Stacked GAN model
gan_input = Input(shape=[100])
gan_V = discriminator(generator(gan_input))
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)



n_train = 20
trainidx = random.sample(range(0, X_train.shape[0]), n_train)
XT = X_train[trainidx,:,:]



noise_gen = numpy.random.uniform(0,1,size=[XT.shape[0],100])
generated_images = generator.predict(noise_gen)
X_pre = r_[XT.reshape((XT.shape[0], 1) + XT.shape[1:]), generated_images]
n = XT.shape[0]
y_pre = zeros([2*n,2])
y_pre[:n,1] = 1
y_pre[n:,0] = 1
make_trainable(discriminator, True)
discriminator.fit(X_pre, y_pre, nb_epoch=1, batch_size=128)
y_hat = discriminator.predict(X_pre)

y_hat_idx = argmax(y_hat, axis=1)
y_idx = argmax(y_pre, axis=1)
diff = y_idx - y_hat_idx
n_tot = y_pre.shape[0]
n_rig = (diff==0).sum()
acc = n_rig*100.0/n_tot
print("Accuracy: {:0.02f}% ({} of {}) right".format(acc, n_rig, n_tot))

def train_for_n(nb_epoch=5000, BATCH_SIZE=32):
    losses = zeros([nb_epoch,2])
    for e in range(nb_epoch):        
        # Make generative images
        image_batch = X_train[numpy.random.randint(0,X_train.shape[0], size=BATCH_SIZE),:,:]
        noise_gen = numpy.random.uniform(0,1,size=[BATCH_SIZE,100])
        generated_images = generator.predict(noise_gen)

        # Train discriminator
        X = r_[image_batch.reshape((image_batch.shape[0], 1) + image_batch.shape[1:]),
               generated_images]
        y = np.zeros([2*BATCH_SIZE, 2])
        y[:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1

        make_trainable(discriminator, True)
        losses[e,0] = discriminator.train_on_batch(X,y)

        # Train generator
        noise_tr = numpy.random.uniform(0,1,size=[BATCH_SIZE,100])
        y2 = zeros([BATCH_SIZE,2])
        y2[:,1] = 1

        make_trainable(discriminator, False)
        losses[e,1] = GAN.train_on_batch(noise_tr, y2)

    return losses

with open("discriminator.json", "w") as f:
    f.write(discriminator.to_json())
with open("generator.json", "w") as f:
    f.write(generator.to_json())

# Train
losses_1 = train_for_n(nb_epoch=6*n_train)
savetxt("mnist_train_1.csv", losses_1)
discriminator.save("discriminator_1.hdf5")
generator.save("generator_1.hdf5")

# Slow down; train
opt.lr.set_value(1e-5)
dopt.lr.set_value(1e-4)
losses_2 = train_for_n(nb_epoch=2*n_train)
savetxt("mnist_train_2.csv", losses_2)
discriminator.save("discriminator_2.hdf5")
generator.save("generator_2.hdf5")

# Slow down; train
opt.lr.set_value(1e-6)
dopt.lr.set_value(1e-5)
losses_3 = train_for_n(nb_epoch=2*n_train)
savetxt("mnist_train_3.csv", losses_3)
discriminator.save("discriminator_3.hdf5")
generator.save("generator_3.hdf5")

print("Finished training")

testidx = random.sample(range(0, X_test.shape[0]), n_train)
XT = X_train[testidx,:,:]

noise_gen = numpy.random.uniform(0,1,size=[XT.shape[0],100])
generated_images = generator.predict(noise_gen)
X_post = r_[XT.reshape((XT.shape[0], 1) + XT.shape[1:]), generated_images]
n = XT.shape[0]
y_post = zeros([2*n,2])
y_post[:n,1] = 1
y_post[n:,0] = 1
y_hat = discriminator.predict(X_post)

y_hat_idx = argmax(y_hat, axis=1)
y_idx = argmax(y_post, axis=1)
diff = y_idx - y_hat_idx
n_tot = y_post.shape[0]
n_rig = (diff==0).sum()
acc = n_rig*100.0/n_tot
print("Accuracy: {:0.02f}% ({} of {}) right".format(acc, n_rig, n_tot))
