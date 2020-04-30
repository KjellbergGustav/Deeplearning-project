import numpy as np
from tensorflow.keras.datasets import mnist
import cv2
from random import randint

class Noises:

    @staticmethod
    def gaussian(trainX, testX):
        # sample noise from a random normal distribution centered at 0.5 (since
        # our images lie in the range [0, 1]) and a standard deviation of 0.5
        trainNoise = np.random.normal(loc=0.5, scale=0.5, size=trainX.shape)
        testNoise = np.random.normal(loc=0.5, scale=0.5, size=testX.shape)
        trainXNoisy = np.clip(trainX + trainNoise, 0, 1)
        testXNoisy = np.clip(testX + testNoise, 0, 1)
        return trainXNoisy, testXNoisy

    @staticmethod
    def speckle(trainX, testX):
        trainNoise = np.random.normal(loc=0, scale=1, size=trainX.shape)
        testNoise = np.random.normal(loc=0, scale=1, size=testX.shape)
        trainXNoisy = np.clip(trainX + trainX * trainNoise, 0, 1)
        testXNoisy = np.clip(testX + testX * testNoise, 0, 1)
        return trainXNoisy, testXNoisy

    @staticmethod
    def salt_and_pepper(trainX, testX):
        # need to produce a copy as to not modify the original image
        trainXcopy = trainX.copy()
        testXcopy = testX.copy()
        row, col, _ = trainXcopy[0].shape
        salt_vs_pepper = 0.2
        amount = 0.3
        num_salt = np.ceil(amount * trainXcopy[0].size * salt_vs_pepper)
        num_pepper = np.ceil(amount * trainXcopy[0].size * (1.0 - salt_vs_pepper))

        # training data
        for x in trainXcopy:
            # add Salt noise
            coords = [np.random.randint(0, i, int(num_salt)) for i in x.shape]
            x[coords[0], coords[1], :] = 1

            # add Pepper noise
            coords = [np.random.randint(0, i, int(num_pepper)) for i in x.shape]
            x[coords[0], coords[1], :] = 0

        # testing data
        for x in testXcopy:
            # add Salt noise
            coords = [np.random.randint(0, i, int(num_salt)) for i in x.shape]
            x[coords[0], coords[1], :] = 1

            # add Pepper noise
            coords = [np.random.randint(0, i, int(num_pepper)) for i in x.shape]
            x[coords[0], coords[1], :] = 0

        return trainXcopy, testXcopy

    @staticmethod
    def block(trainX, testX):
        trainXcopy = trainX.copy()
        testXcopy = testX.copy()

        for x in testXcopy:
            end = x.shape[0] - int(x.shape[0] * 0.4)
            coords1 = randint(0, end)
            coords2 = randint(0, end)
            coords3 = coords1 + int(x.shape[0] * 0.4)
            coords4 = coords2 + int(x.shape[0] * 0.4)
            x[coords1:coords3,coords2:coords4] = 1
            
        return trainXcopy, testXcopy
        
    @staticmethod
    def border(trainX, testX):
        trainXcopy = trainX.copy()
        testXcopy = testX.copy()

        for x in trainXcopy:
            a = int(x.shape[0] // 10)
            b = x.shape[0] - a
            x[:a,:] = 1
            x[b:,:] = 1
            x[:,:a] = 1
            x[:,b:] = 1

        for x in testXcopy:
            a = int(x.shape[0] // 10)
            b = x.shape[0] - a
            x[:a,:] = 1
            x[b:,:] = 1
            x[:,:a] = 1
            x[:,b:] = 1

        return trainXcopy, testXcopy

def save_images(testX, testXNoisy, file_path):
    outputs = None
    # loop over our number of output samples
    for i in range(0, 8):
        # grab the original image and reconstructed image
        noise = (testXNoisy[i] * 255).astype("uint8")
        noNoise = (testX[i] * 255).astype("uint8")
        
        # stack the original and reconstructed image side-by-side
        output = np.hstack([noise, noNoise])
        
        # if the outputs array is empty, initialize it as the current
        # side-by-side image display
        if outputs is None:
            outputs = output

        # otherwise, vertically stack the outputs
        else:
            outputs = np.vstack([outputs, output])

    # save the outputs image to disk
    cv2.imwrite(file_path, outputs)


if __name__ == '__main__':
    print("[INFO] loading MNIST dataset...")
    ((trainX, _), (testX, _)) = mnist.load_data()

    # add a channel dimension to every image in the dataset, then scale
    # the pixel intensities to the range [0, 1]
    trainX = np.expand_dims(trainX, axis=-1)
    testX = np.expand_dims(testX, axis=-1)
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0
    
    # create images with all noises
    trainXNoisy, testXNoisy = Noises.gaussian(trainX, testX)
    save_images(testX, testXNoisy, "noise/gaussianNoise.png")

    trainXNoisy, testXNoisy = Noises.speckle(trainX, testX)
    save_images(testX, testXNoisy, "noise/speckleNoise.png")

    trainXNoisy, testXNoisy = Noises.salt_and_pepper(trainX, testX)
    save_images(testX, testXNoisy, "noise/saltAndPepper.png")
    
    trainXNoisy, testXNoisy = Noises.block(trainX, testX)
    save_images(testX, testXNoisy, "noise/block.png")

    trainXNoisy, testXNoisy = Noises.border(trainX, testX)
    save_images(testX, testXNoisy, "noise/border.png")


