# Counterfactual_synthesis_for_lesion_segmentation

This project aims at segmenting multiple sclerosis lesions in the Spinal Cord on MRI images. 
The main idea consists in training model on healthy patients that is able to recover the initial image after degradation. Then using it on images with lesions and segment the lesion based on the differences between original and recovered image.

## Simple 2D diffusion model

This first model, in it's dedicated branch implement a 2D diffusion model without a latent splace

## Simple 3D Autoencoder

Tries to reconstruct the IRM data with a simple 3D autoencoder

### Usage

In order to launch the training, one can use this command :
`python train.py --evaluate True --model_path /path/to/model.pth --model_output path/to/model_out.pth --data_path path/to/data-multi-subject/ --num_epochs 200 --batch_size 1  --learning_rate 0.001'


### Dataset

The original training set is extracted from the public dataset [Spine Generic](https://github.com/spine-generic/data-multi-subject).
This Dataset contains 3D images and require proper prepocessing to be used.
The spine generic dataset is meant to be stored like data/data-multi-subject/..

### Preprocessing

The data has been preprocesse to make the network as robust as possible

* The dataset is filtered to keep T1w and T2w images paths only.
* The dataset is splited between train patients and test patients (20% test)
* Two object from the class "2D_dataset" are created. They encapsulate the label and the image path.
* At each training epoch, the model sees each 3D image once. Each time the image is randomly :
    - flipped
    - rotated (in a 3° range)
    - Shifted (in a 0.1 range)
    - reframed (in a 2D fashion, with minimum size (30 * 30))

### Network Structure

The network used is a simple convolutional auto-encoder defined in Auto_encoder_network.py