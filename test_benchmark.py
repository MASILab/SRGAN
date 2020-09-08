import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_30.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}, 'image_name_test': {'psnr': [], 'ssim': []}}

model = Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('/home/local/VANDERBILT/kanakap/PycharmProjects/CDMRI-SRGAN/SRGAN/epochs/' + MODEL_NAME))

test_set = TestDatasetFromFolder('/home/local/VANDERBILT/kanakap/test_info.csv', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
sr_images = []

for image_name, lr_image, hr_restore_img, hr_image, pad_idx, orig_shape in test_bar:
    print('image name', image_name)
    image_name = image_name[0]
    lr_image = Variable(lr_image, volatile=True)
    hr_image = Variable(hr_image, volatile=True)
    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()

    sr_image = model(lr_image)
    sr_images.append(sr_image.cpu())
    mse = ((hr_image - sr_image) ** 2).data.mean()
    psnr = 10 * log10(1 / mse)
    print(psnr)
    print(sr_image.shape, hr_image.shape)
    ssim = pytorch_ssim.ssim(sr_image, hr_image)
    print(ssim)

    #test_images = torch.stack(
        #[display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
         #display_transform()(sr_image.data.cpu().squeeze(0))])
        #[hr_restore_img.squeeze(0), hr_image.data.cpu().squeeze(0),
         #sr_image.data.cpu().squeeze(0)])



    #image = utils.make_grid(test_images, nrow=3, padding=5)
    #utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                     #image_name.split('.')[-1], padding=5)

    # save psnr\ssim
    results['image_name_test']['psnr'].append(psnr)
    results['image_name_test']['ssim'].append(ssim)

out_path = '/home/local/VANDERBILT/kanakap/PycharmProjects/CDMRI-SRGAN/SRGAN/statistics/'
saved_results = {'psnr': [], 'ssim': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    if (len(psnr) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

lx, lX, ly, lY, lz, lZ, rx, rX, ry, rY, rz, rZ = pad_idx
print(len(sr_images))
out = torch.cat(sr_images, dim=0).squeeze()
print(out.shape)
out = out.permute(1, 2, 3, 0).detach().numpy()
final_out = np.zeros([80, 92, 56, 2])
final_out[rx:rX, ry:rY, rz:rZ] = out[lx:lX, ly:lY, lz:lZ]
#, rz:rZ]
ref = nib.load(image_name)

nii = nib.Nifti1Image(final_out, affine=ref.affine, header=ref.header)
nib.save(nii, out_path + 'image_result__psnr_%.4f_ssim_%.4f.nii.gz' % (psnr, ssim))

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')
