import argparse
import time

import nibabel as nib
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_30.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

def pad(img):
    tmp = np.zeros((96, 96, 96, img.shape[3]))

    diff = int((96-img.shape[0])/2)
    lx = max(diff,0)
    lX = min(img.shape[0]+diff,96)

    diff = (img.shape[0]-96) / 2
    rx = max(int(np.floor(diff)),0)
    rX = min(img.shape[0]-int(np.ceil(diff)),img.shape[0])

    diff = int((96 - img.shape[1]) / 2)
    ly = max(diff, 0)
    lY = min(img.shape[1] + diff, 96)

    diff = (img.shape[1] - 96) / 2
    ry = max(int(np.floor(diff)), 0)
    rY = min(img.shape[1] - int(np.ceil(diff)), img.shape[1])

    diff = int((96 - img.shape[2]) / 2)
    lz = max(diff, 0)
    lZ = min(img.shape[2] + diff, 96)

    diff = (img.shape[2] - 96) / 2
    rz = max(int(np.floor(diff)), 0)
    rZ = min(img.shape[2] - int(np.ceil(diff)), img.shape[2])

    tmp[lx:lX,ly:lY,lz:lZ] = img[rx:rX,ry:rY,rz:rZ]

    return tmp, [lx, lX, ly, lY, lz, lZ, rx, rX, ry, rY, rz, rZ]

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('/home/local/VANDERBILT/kanakap/PycharmProjects/CDMRI-SRGAN/SRGAN/epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('/home/local/VANDERBILT/kanakap/PycharmProjects/CDMRI-SRGAN/SRGAN/epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

# image = Image.open(IMAGE_NAME)
# image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)

nii_lr = nib.load(IMAGE_NAME)
lr_image = nii_lr.get_fdata()
# pad the surrounding
lr_image_pad, pad_idx = pad(lr_image)

vol_idx = np.random.randint(0, lr_image.shape[3])
lr_vol = torch.from_numpy(lr_image_pad[:, :, :, vol_idx]).unsqueeze(0).float()

# normalize the data
lr_max = np.percentile(lr_vol, 99.99)
lr_vol = lr_vol / lr_max
print(lr_vol.size())

if TEST_MODE:
    lr_vol = lr_vol.cuda()

start = time.clock()
out = model(lr_vol)
elapsed = (time.clock() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out[0].data.cpu())
out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
