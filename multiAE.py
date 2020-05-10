from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model, load_model
from convautoencoder import ConvAutoencoder
import os.path
import h5py



class MultiAE:
    def __init__(self):
        model = None

    def build(self, height, width, depth, filters=(32, 64), latentDim=16):
        inputShape = (height, width, depth)
        inputs = Input(shape=inputShape)
        cur_path = os.path.dirname(__file__)
        macPath = os.path.join(cur_path,'models/sig_last_final_')
        #winPath = os.path.join(cur_path, '.\models\');
        path = macPath
        # loading existing model
        print ("Loading existing autoencoders...")

        file_path = path+'convnetDenoisingAEgaussian.h5'
        if os.path.isfile(file_path):
                cae = ConvAutoencoder()
                autoencoder_gaussian = cae.load_model(file_path)
        file_path = path+'convnetDenoisingAEspeckle.h5'
        if os.path.isfile(file_path):
                cae = ConvAutoencoder()
                autoencoder_speckle = cae.load_model(file_path)
        file_path = path+'convnetDenoisingAEsaltAndPepper.h5'
        if os.path.isfile(file_path):
                cae = ConvAutoencoder()
                autoencoder_saltAndPepper = cae.load_model(file_path)
        file_path = path+'convnetDenoisingAEblock.h5'
        if os.path.isfile(file_path):
                cae = ConvAutoencoder()
                autoencoder_block = cae.load_model(file_path)
        file_path = path+'convnetDenoisingAEborder.h5'
        if os.path.isfile(file_path):
                cae = ConvAutoencoder()
                autoencoder_border = cae.load_model(file_path)
        file_path = path+'convnetDenoisingAEnoNoise.h5'
        if os.path.isfile(file_path):
                cae = ConvAutoencoder()
                autoencoder_no_noise = cae.load_model(file_path)

        #
        x_gaussian = autoencoder_gaussian(inputs)
        x_speckle = autoencoder_speckle(inputs)
        x_saltAndPepper = autoencoder_saltAndPepper(inputs)
        x_block = autoencoder_block(inputs)
        x_border = autoencoder_border(inputs)
        x_noNoise = autoencoder_no_noise(inputs)


        # concatinating autoencoders
        combined = concatenate([x_gaussian, x_speckle, x_saltAndPepper, x_block, x_border, x_noNoise])

        # reshaping to fit input/output
        combined = Dense(1, input_shape=(6,))(combined)
        #combined = Reshape((28,28,1))(combined)

        # apply RELU => LINEAR (Don't know if this is needed)
        x = Activation("relu")(combined)
        x = Activation("tanh")(x)
        
        # out models is the combined autoencoders 
        self.model = Model(inputs=inputs, outputs=x)
        
        return self.model

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = load_model(file_path)
        return self.model