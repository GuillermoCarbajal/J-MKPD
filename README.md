# J-MKPD

Official Pytorch Implementation  of *Blind Motion Deblurring with Pixel-Wise Kernel
Estimation via Kernel Prediction Networks*    

The proposed motion deblurring method generalizes well and produces sharp restorations on real images, in some cases even sharpen than *ground-truth* images. The commonly used quality metric PSNR favors blurry outputs. We show that by increasing the PSRN of the restoration by blurring our deblurring results. Metrics that measure the sharpness of an image (such as CPBD) correlate well with our perception of sharpness.
<p align="center">
<img width="700" src="imgs/realblur_results.png?raw=true">
</p>

## Kernels Prediction Network Architecture
<p align="center">
<img width="900" src="imgs/architecture.png?raw=true">
  </p>
  
## Quick Demo


* <a href="https://colab.research.google.com/github/GuillermoCarbajal/J-MKPD/blob/main/J-MKPD_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Installation
### Clone Repository
```
git clone https://github.com/GuillermoCarbajal/J-MKPD.git
```


### Download deblurring models

[Kernels Prediction Model](https://iie.fing.edu.uy/~carbajal/IEEE_CI_models/COCO900_restL2_sat/80000_kernels_network.pth)           
[Restoration Network](https://iie.fing.edu.uy/~carbajal/IEEE_CI_models/COCO900_restL2_sat/80000_G.pth)

### Deblur an image or a list of images
```
python image_deblurring.py -b blurry_img_path --reblur_model reblur_model_path --nimbusr_model restoration_model_path --output_folder results
```

### Parameters
Additional options:   
  `--blurry_images`: may be a singe image path or a .txt with a list of images.
  
  `--resize_factor`: input image resize factor (default 1)     
  
  `--gamma_factor`: gamma correction factor. By default is assummed `gamma_factor=2.2`. For Kohler dataset images `gamma_factor=1.0`.
  


### Compute kernels from an image

```
python compute_kernels.py -i image_path -m kernels_prediction_model_path
```
Our method generalize better to datasets not seen during training. Other methods motion fields are correlated with the image structure, suffer from the aperture problem and predict deltas on low variance regions.
<p align="center">
<img width="900" src="imgs/kernels_Kohler.png?raw=true">
  </p>


### Saturated images examples

[Kernels Prediction Model (light streaks)](https://iie.fing.edu.uy/~carbajal/IEEE_CI_models/COCO900_restL2_sat_streaks/80000_kernels_network.pth)           
[Restoration Network (light streaks)](https://iie.fing.edu.uy/~carbajal/IEEE_CI_models/COCO900_restL2_sat_streaks/80000_G.pth)


<p align="center">
<img width="900" src="imgs/sat_images_examples.png?raw=true">
  </p>

## Aknowledgments 
We thank the authors of [Deep Model-Based Super-Resolution with Non-Uniform Blur](https://arxiv.org/abs/2204.10109) for the Blind Deconvolution Network provided in https://github.com/claroche-r/DMBSR 


Guillermo Carbajal was supported partially by Agencia Nacional de Investigacion e Innovación (ANII, Uruguay) ´grant POS FCE 2018 1 1007783 and PV by the MICINN/FEDER UE project under Grant PGC2018- 098625-B-I0; H2020-MSCA-RISE-2017 under Grant 777826 NoMADS and Spanish Ministry of Economy and Competitiveness under the Maria de Maeztu Units of Excellence Programme (MDM-2015-0502). The experiments presented in this paper were carried out using ClusterUY (site: https://cluster.uy) and GPUs donated by NVIDIA Corporation.
