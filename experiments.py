# import the necessary packages
import os.path
from multiAE import MultiAE
from noises import Noises
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os.path
import os
from convautoencoder import ConvAutoencoder

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# load the MNIST dataset
print("[INFO] loading MNIST dataset...")
((trainX, _), (testX, _)) = mnist.load_data()

# add a channel dimension to every image in the dataset, then scale
# the pixel intensities to the range [0, 1]
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

numberOfTrainX = len(trainX)//6
numberOfTestX = len(testX)//6

trainXNoisyGaussian, testXNoisyGaussian = Noises.gaussian(trainX[:numberOfTrainX], testX[:numberOfTestX])
trainXNoisySpeckle, testXNoisySpeckle = Noises.speckle(trainX[numberOfTrainX:numberOfTrainX*2], testX[numberOfTestX:numberOfTestX*2])
trainXNoisySaltAndPepper, testXNoisySaltAndPepper = Noises.salt_and_pepper(trainX[numberOfTrainX*2:numberOfTrainX*3], testX[numberOfTestX*2:numberOfTestX*3])
trainXNoisyBlock, testXNoisyBlock = Noises.block(trainX[numberOfTrainX*3:numberOfTrainX*4], testX[numberOfTestX*3:numberOfTestX*4])
trainXNoisyBorder, testXNoisyBorder = Noises.border(trainX[numberOfTrainX*4:numberOfTrainX*5], testX[numberOfTestX*4:numberOfTestX*5])
trainXNoisyNoNoise, testXNoisyNoNoise = trainX[numberOfTrainX*5:], testX[numberOfTestX*5:]

trainXNoisy = np.concatenate((trainXNoisyGaussian, trainXNoisySpeckle, trainXNoisySaltAndPepper, trainXNoisyBlock, trainXNoisyBorder, trainXNoisyNoNoise))
testXNoisy = np.concatenate((testXNoisyGaussian, testXNoisySpeckle, testXNoisySaltAndPepper, testXNoisyBlock, testXNoisyBorder, testXNoisyNoNoise))


# initialize convolutional autoencoder
mnn = MultiAE()

# If mac use below
cur_path = os.path.dirname(__file__)
# ------ DnCNN -------
#file_path = os.path.join(cur_path,'models/sig_last_final_best_multi.h5')

# ------ AE all noises -------
file_path = os.path.join(cur_path,'models/sig_final_best_multiAE_tan.h5')

# If windows use below
#file_path = '.\models\multiAE.h5'
# loading existing model

print ("Loading existing model...")
autoencoder_mcdncnn = mnn.load_model(file_path)

cae = ConvAutoencoder()

file_path = os.path.join(cur_path,'models/sig_last_final_convnetDenoisingAEallNoises.h5')

cae_allnoises = cae.load_model(file_path)






#BatchSizes = [32, 64, 128, 256, 512, 1024]
gaussSum = 0
speckleSum = 0
s_pSum = 0
blockSum = 0
borderSum = 0
noNoiseSum = 0

cae_loss = []
dncnn_loss = []
#np.random.shuffle(testXNoisy)

p = np.random.permutation(len(testX))
testX = testX[p] 
testXNoisy = testXNoisy[p]
for indexFactor in range(100):

    startIndex = indexFactor*100
    endIndex = (indexFactor+1)*100

    testSplit = testXNoisy[startIndex:endIndex]
    labelSplit = testX[startIndex:endIndex]
    print(indexFactor)
    loss_cae = cae_allnoises.evaluate(testSplit, labelSplit, batch_size=32, verbose=0)

    cae_loss.append(loss_cae)

    loss_mcdncnn = autoencoder_mcdncnn.evaluate(testSplit, labelSplit, batch_size=32, verbose=0)

    dncnn_loss.append(loss_mcdncnn)

mean_mc = np.mean(dncnn_loss)
std_mc = np.std(dncnn_loss)
print("mean McDnCnn: ", np.mean(dncnn_loss))
print("std McDnCnn: ", np.std(dncnn_loss))

mean_cae = np.mean(cae_loss)
std_cae = np.std(cae_loss)
print("mean CAE: ", np.mean(cae_loss))
print("std CAE: ", np.std(cae_loss))


p_top = mean_mc-mean_cae
p_bot = np.sqrt((np.square(std_cae)/100)+(np.square(std_mc)/100))
p = p_top/p_bot

print("P-value: ", p)




"""for BS in BatchSizes:
    


    # evalute model on different noises
    _, testXNoisy = Noises.gaussian(trainX, testX)
    print("Evaluate model on Gaussian noise:")
    [lossGauss, acc] = autoencoder.evaluate(testXNoisy, testX, batch_size=BS)
    print(acc)
    gaussSum += lossGauss

    _, testXNoisy = Noises.speckle(trainX, testX)
    print("Evaluate model on Speckle noise:")
    lossSpeckle = autoencoder.evaluate(testXNoisy, testX, batch_size=BS)
    speckleSum += lossSpeckle

    _, testXNoisy = Noises.salt_and_pepper(trainX, testX)
    print("Evaluate model on Salt&Pepper noise:")
    lossS_P = autoencoder.evaluate(testXNoisy, testX, batch_size=BS)
    s_pSum += lossS_P

    _, testXNoisy = Noises.block(trainX, testX)
    print("Evaluate model on Block noise:")
    lossBlock = autoencoder.evaluate(testXNoisy, testX, batch_size=BS)
    blockSum += lossBlock

    _, testXNoisy = Noises.border(trainX, testX)
    print("Evaluate model on Border noise:")
    lossBorder = autoencoder.evaluate(testXNoisy, testX, batch_size=BS)
    borderSum += lossBorder

    testXNoisy = np.copy(testX)
    print("Evaluate model on no noise:")
    lossNone = autoencoder.evaluate(testXNoisy, testX, batch_size=BS)
    noNoiseSum += lossNone

    print('Loss: Gauss, speckle, S_P, Block, Border, None')
    print('Loss: ', lossGauss, " ", lossSpeckle, " ", lossS_P, " ", lossBlock, " ", lossBorder, " ", lossNone)

print("Averge loss over 6 runs with batch sizes: ", BatchSizes, " and a data set size of", testX.shape)

print("Gauss: ", gaussSum/len(BatchSizes))
print("Speckle: ", speckleSum/len(BatchSizes))
print("Salt and Pepper: ", s_pSum/len(BatchSizes))
print("Block: ", blockSum/len(BatchSizes))
print("Border: ", borderSum/len(BatchSizes))
print("No noise: ", noNoiseSum/len(BatchSizes))
sumAll = gaussSum/len(BatchSizes)+ speckleSum/len(BatchSizes)+s_pSum/len(BatchSizes)+blockSum/len(BatchSizes)+borderSum/len(BatchSizes)+noNoiseSum/len(BatchSizes)
print("Joint average: ", sumAll/6)"""