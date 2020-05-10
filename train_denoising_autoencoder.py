# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import os.path
from convautoencoder import ConvAutoencoder
from noises import Noises
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import argparse
import cv2
import os.path
import os

# Necessary for it to work due to some weird issue...
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="outputs/output.png",
    help="path to output visualization file")
ap.add_argument("-p", "--plot", type=str, default="plots/plot.png",
    help="path to output plot file")
ap.add_argument("-n", "--noise", type=str, default="gaussian",
    help="type of noise to add")
args = vars(ap.parse_args())

# initialize the number of epochs to train for and batch size
EPOCHS = 25
BS = 32

# load the MNIST dataset
print("[INFO] loading MNIST dataset...")
((trainX, _), (testX, _)) = mnist.load_data()

# add a channel dimension to every image in the dataset, then scale
# the pixel intensities to the range [0, 1]
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0


if args["noise"] == "gaussian":
    trainXNoisy, testXNoisy = Noises.gaussian(trainX, testX)
elif args["noise"] == "speckle":
    trainXNoisy, testXNoisy = Noises.speckle(trainX, testX)
elif args["noise"] == "saltAndPepper":
    trainXNoisy, testXNoisy = Noises.salt_and_pepper(trainX, testX)
elif args["noise"] == "block":
    trainXNoisy, testXNoisy = Noises.block(trainX, testX)
elif args["noise"] == "border":
    trainXNoisy, testXNoisy = Noises.border(trainX, testX)
elif args["noise"] == "noNoise":
    trainXNoisy = np.copy(trainX)
    testXNoisy = np.copy(testX)
elif args["noise"] == "allNoises":
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
cae = ConvAutoencoder()

# If mac, use below
cur_path = os.path.dirname(__file__)
file_path = os.path.join(cur_path,'models/sig_last_final_convnetDenoisingAE'+ args["noise"]+ '.h5')

# If windows, use below
#file_path = '.\models\convnetDenoisingAE' + args["noise"] + '.h5'


if os.path.isfile(file_path):
        # loading existing model
        print ("Loading existing model...")
        autoencoder = cae.load_model(file_path)
else:
    # construct our convolutional autoencoder
    print("[INFO] building autoencoder...")
    autoencoder = cae.build(28, 28, 1, args["noise"])
    opt = Adam(lr=1e-3)
    autoencoder.compile(loss="mse", optimizer=opt)
    mc = ModelCheckpoint(file_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # train the convolutional autoencoder
    H = autoencoder.fit(
        trainXNoisy, trainX,
        validation_data=(testXNoisy, testX),
        epochs=EPOCHS,
        batch_size=BS,
        callbacks=[mc])
    # save model
    cae.save_model(file_path)

    # construct a plot that plots and saves the training history
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])

# use the convolutional autoencoder to make predictions on the
# testing images, then initialize our list of output images
print("[INFO] making predictions...")
decoded = autoencoder.predict(testXNoisy)
outputs = None

lst = list(range(len(decoded)))
for i in lst[0::500]:
    # grab the original image and reconstructed image
    original = (testXNoisy[i] * 255).astype("uint8")
    recon = (decoded[i] * 255).astype("uint8")
    noNoise = (testX[i] * 255).astype("uint8")
    
    # stack the original and reconstructed image side-by-side
    output = np.hstack([original, recon, noNoise])
    
    # if the outputs array is empty, initialize it as the current
    # side-by-side image display
    if outputs is None:
        outputs = output

    # otherwise, vertically stack the outputs
    else:
        outputs = np.vstack([outputs, output])

# save the outputs image to disk
cv2.imwrite(args["output"], outputs)


# evalute model on different noises
_, testXNoisy = Noises.gaussian(trainX, testX)
print("Evaluate model on Gaussian noise:")
lossGauss = autoencoder.evaluate(testXNoisy, testX, batch_size=BS)


_, testXNoisy = Noises.speckle(trainX, testX)
print("Evaluate model on Speckle noise:")
lossSpeckle = autoencoder.evaluate(testXNoisy, testX, batch_size=BS)

_, testXNoisy = Noises.salt_and_pepper(trainX, testX)
print("Evaluate model on Salt&Pepper noise:")
lossS_P = autoencoder.evaluate(testXNoisy, testX, batch_size=BS)

_, testXNoisy = Noises.block(trainX, testX)
print("Evaluate model on Block noise:")
lossBlock = autoencoder.evaluate(testXNoisy, testX, batch_size=BS)

_, testXNoisy = Noises.border(trainX, testX)
print("Evaluate model on Border noise:")
lossBorder = autoencoder.evaluate(testXNoisy, testX, batch_size=BS)

testXNoisy = np.copy(testX)
print("Evaluate model on no noise:")
lossNone = autoencoder.evaluate(testXNoisy, testX, batch_size=BS)

print('Loss: Gauss, speckle, S_P, Block, Border, None')
print('Loss: ', lossGauss, " ", lossSpeckle, " ", lossS_P, " ", lossBlock, " ", lossBorder, " ", lossNone)