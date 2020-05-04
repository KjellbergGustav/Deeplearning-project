# Deeplearning-project
#### To run train/ test the autoencoders, run:

###### python train_denoising_autoencoder.py --sample 8 --output outputs/output.png --plot plots/plot.png --noise gaussian

This will result in the training of an autoencoder which denoises gaussian noise. 8 sample images will be saves in the file outputs/output.png and a plot of the training loss/accuray will be saved in the file plots/plot.png. The available noises are gaussian, speckle, saltAndPepper, block, border and noNoise. Only gaussian and speckle are pre-trained, to run the other noises will require training which takes about 45 minutes.