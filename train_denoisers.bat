python train_denoising_autoencoder.py --output outputs/output_denoising_gaussian.png --plot plots/plot_denoising_gaussian.png --noise gaussian
python train_denoising_autoencoder.py --output outputs/output_denoising_speckle.png --plot plots/plot_denoising_speckle.png --noise speckle
python train_denoising_autoencoder.py --output outputs/output_denoising_saltAndPepper.png --plot plots/plot_denoising_saltAndPepper.png --noise saltAndPepper
python train_denoising_autoencoder.py --output outputs/output_denoising_block.png --plot plots/plot_denoising_block.png --noise block
python train_denoising_autoencoder.py --output outputs/output_denoising_border.png --plot plots/plot_denoising_border.png --noise border
python train_denoising_autoencoder.py --output outputs/output_denoising_noNoise.png --plot plots/plot_denoising_noNoise.png --noise noNoise
python train_denoising_autoencoder.py --output outputs/output_denoising_alNoises.png --plot plots/plot_denoising_allNoises.png --noise allNoises