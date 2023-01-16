import numpy as np
import argparse
from models.TwoHeadsNetwork import TwoHeadsNetwork
from models.network_nimbusr import NIMBUSRforSI as netForSI
from models.network_nimbusr import NIMBUSR as net

import torch

from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import gray2rgb


from torchvision import transforms
import os
import json

from utils.visualization import save_image, tensor2im, save_kernels_grid


def load_nimbusr_net(type='nimbusr'):
    opt_net = { "n_iter": 8
        , "h_nc": 64
        , "in_nc": 4
        , "out_nc": 3
        , "ksize": 25
        , "nc": [64, 128, 256, 512]
        , "nb": 2
        , "gc": 32
        , "ng": 2
        , "reduction" : 16
        , "act_mode": "R" 
        , "upsample_mode": "convtranspose" 
        , "downsample_mode": "strideconv"}

    opt_data = { "phase": "train"
          , "dataset_type": "usrnet_multiblur"
          , "dataroot_H": "datasets/COCO/val2014"
          , "dataroot_L": None
          , "H_size": 256
          , "use_flip": True
          , "use_rot": True
          , "scales": [2]
          , "sigma": [0, 2]
          , "sigma_test": 15
          , "n_channels": 3
          , "dataloader_shuffle": True
          , "dataloader_num_workers": 16
          , "dataloader_batch_size": 16
          , "motion_blur": True

          , "coco_annotation_path": "datasets/COCO/instances_val2014.json"}

    path_pretrained = args.nimbusr_model #r'../model_zoo/NIMBUSR.pth'
    
    if type=='nimbusr':
        netG = net(n_iter=opt_net['n_iter'],
                    h_nc=opt_net['h_nc'],
                    in_nc=opt_net['in_nc'],
                    out_nc=opt_net['out_nc'],
                    nc=opt_net['nc'],
                    nb=opt_net['nb'],
                    act_mode=opt_net['act_mode'],
                    downsample_mode=opt_net['downsample_mode'],
                    upsample_mode=opt_net['upsample_mode']
                    )
    elif type=='nimbusr_sat':
        netG = netForSI(n_iter= opt_net['n_iter'],
                    h_nc=opt_net['h_nc'],
                    in_nc=opt_net['in_nc'],
                    out_nc=opt_net['out_nc'],
                    nc=opt_net['nc'],
                    nb=opt_net['nb'],
                    act_mode=opt_net['act_mode'],
                    downsample_mode=opt_net['downsample_mode'],
                    upsample_mode=opt_net['upsample_mode']
                    )

    netG.load_state_dict(torch.load(path_pretrained))
    netG = netG.to('cuda')

    return netG


parser = argparse.ArgumentParser()
parser.add_argument('--blurry_images', '-b', type=str, required=True, help='list with the original blurry images or path to a blurry image')
parser.add_argument('--reblur_model', '-m', type=str, required=True, help='two heads reblur model')
parser.add_argument('--gpu_id', '-g', type=int, default=0)
parser.add_argument('--output_folder','-o', type=str, help='output folder', default='testing_results')
parser.add_argument('--resize_factor','-rf', type=float, default=1)
parser.add_argument('--gamma_factor', type=float, default=2.2, help='gamma correction factor')
parser.add_argument('--nimbusr_model','-nm', type=str, default=r'../model_zoo/NIMBUSR.pth')
parser.add_argument('--network_type','-nt', type=str, default=r'nimbusr')




'''
-b /media/carbajal/OS/data/datasets/cvpr16_deblur_study_real_dataset/real_dataset/coke.jpg  -m /media/carbajal/OS/data/models/ade_dataset/NoFC/gamma_correction/L1/L2_epoch150_epoch150_L1_epoch900.pkl -n 20  --saturation_method 'combined'
'''

args = parser.parse_args()

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

with open(os.path.join(args.output_folder, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

def get_images_list(list_path):

    with open(list_path) as f:
        images_list = f.readlines()
        images_list = [l[:-1] for l in images_list]
    f.close()

    return images_list


if args.blurry_images.endswith('.txt'):
    blurry_images_list = get_images_list(args.blurry_images)
elif (args.blurry_images.endswith('.png') or args.blurry_images.endswith('.jpg')) :
    blurry_images_list = [args.blurry_images]
else:
    blurry_images_list = os.listdir(args.blurry_images)
    blurry_images_list = [os.path.join(args.blurry_images, f) for f in blurry_images_list]


K=25
two_heads = TwoHeadsNetwork(K).cuda(args.gpu_id)   
two_heads.load_state_dict(torch.load(args.reblur_model, map_location='cuda:%d' % args.gpu_id))
two_heads.eval()


for i,blurry_path in enumerate(blurry_images_list):

    img_name, ext = blurry_path.split('/')[-1].split('.')
    blurry_image =  imread(blurry_path)
    if len(blurry_image.shape) > 2:
        blurry_image = blurry_image[:,:,:3]
    else:
        blurry_image = np.concatenate((blurry_image[:,:,None],blurry_image[:,:,None],blurry_image[:,:,None]), axis=2)

    M, N, C = blurry_image.shape
    if args.resize_factor != 1:
        if len(blurry_image.shape) == 2:
            blurry_image = gray2rgb(blurry_image)
        new_shape = (int(args.resize_factor*M), int(args.resize_factor*N), C )
        blurry_image = resize(blurry_image,new_shape, anti_aliasing=True).astype(np.float32)


    initial_image = blurry_image.copy()

    blurry_tensor = transforms.ToTensor()(blurry_image)
    blurry_tensor = blurry_tensor[None,:,:,:]
    blurry_tensor = blurry_tensor.cuda(args.gpu_id)

    save_image(tensor2im(blurry_tensor[0] - 0.5), os.path.join(args.output_folder,
                                                       img_name + '.png' ))

    with torch.no_grad():
        blurry_tensor_to_compute_kernels = blurry_tensor**args.gamma_factor - 0.5
        kernels, masks = two_heads(blurry_tensor_to_compute_kernels)
        save_kernels_grid(blurry_tensor[0],kernels[0], masks[0], os.path.join(args.output_folder, img_name + '_kernels'+'.png'))


    noise_level = 0.01
    with torch.no_grad():
        netG = load_nimbusr_net(args.network_type)
        noise_level = torch.FloatTensor([noise_level]).view(1,1,1).cuda(args.gpu_id)
        kernels_flipped = torch.flip(kernels, dims=(2, 3))
        output = netG(blurry_tensor, masks, kernels_flipped, 1, sigma=noise_level[None,:])
        print('Restored image range:', output.min(), output.max())

    #output_img = tensor2im(torch.clamp(output[0]/output[0].max(),0,1) - 0.5)
    output_img = tensor2im(torch.clamp(output[0],0,1) - 0.5)
    save_image(output_img, os.path.join(args.output_folder, img_name  + '_restored.png' ))
    print('Output saved in ', os.path.join(args.output_folder, img_name  + '_restored.png' ))

