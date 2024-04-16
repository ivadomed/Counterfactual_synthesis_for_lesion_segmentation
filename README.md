# Counterfactual_synthesis_for_lesion_segmentation
 

This project aims at segmenting multiple sclerosis lesions in the Spinal Cord on MRI images. 
The main idea consists in training model on healthy patients that is able to recover the initial image after degradation. Then using it on images with lesions and segment the lesion based on the differences between original and recovered image.

## Simple 2D diffusion model

This first model, in it's dedicated branch implement a 2D diffusion model without a latent splace.