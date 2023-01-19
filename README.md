# J-MKPD

Official Pytorch Implementation  of Blind Motion Deblurring with Pixel-Wise Kernel
Estimation via Kernel Prediction Networks [<a href="https://arxiv.org/abs/2102.01026">ArXiv</a>]
<p align="center">
<img width="700" src="imgs/realblur_results.png?raw=true">
</p>

## Kernels Prediction Network Architecture
<p align="center">
<img width="900" src="imgs/architecture.png?raw=true">
  </p>
  
## Getting Started

### Clone Repository
```
git clone https://github.com/GuillermoCarbajal/J-MKPD.git
```

### Download the pretrained model

Model can be downloaded from [here](https://www.dropbox.com/s/ro9smg1i7lh5b8d/TwoHeads.pkl?dl=0)   

### Compute kernels from an image
```
python compute_kernels.py -i image_path -m model_path
```


### Deblur an image or a list of images
```
python image_deblurring.py -b blurry_img_path --reblur_model reblur_model_path --nimbusr_model restoration_model_path --output_folder results
```

### Parameters
Additional options:   
  `--blurry_images`: may be a singe image path or a .txt with a list of images.
  
  `--resize_factor`: input image resize factor (default 1)     
  
  `--gamma_factor`: gamma correction factor. By default is assummed `gamma_factor=2.2`. For Kohler dataset images `gamma_factor=1.0`.
  

```
## Aknowledgments 

GC was supported partially by Agencia Nacional de Investigacion e Innovación (ANII, Uruguay) ´grant POS FCE 2018 1 1007783 and PV by the MICINN/FEDER UE project under Grant PGC2018- 098625-B-I0; H2020-MSCA-RISE-2017 under Grant 777826 NoMADS and Spanish Ministry of Economy and Competitiveness under the Maria de Maeztu Units of Excellence Programme (MDM-2015-0502). The experiments presented in this paper were carried out using ClusterUY (site: https://cluster.uy) and GPUs donated by NVIDIA Corporation. We also thanks Juan F. Montesinos for his help during the experimental phase.
