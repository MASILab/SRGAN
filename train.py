import argparse
import os
from math import log10

import nibabel as nib
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=30, type=int, help='train epoch number')

# train and valid set
# input_data = pd.read_csv('/nfs/masi/hansencb/CDMRI_2020/challenge_info.csv')
# print(input_data['ISOTROPIC'])

if __name__ == '__main__':
    #torch.backends.cudnn.enabled = False
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    train_set = TrainDatasetFromFolder('/home/local/VANDERBILT/kanakap/challenge_info.csv', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('/home/local/VANDERBILT/kanakap/validation_info.csv', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=1, shuffle=False)
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)
    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = GeneratorLoss()
    
    if torch.cuda.is_available():
        print(netG)
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())
    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    gen_image = 1
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()

        for data, target in train_bar:
            print(data.shape, target.shape)
            g_update_first = True
            batch_size = data.size(0)
            #print('batch_size', batch_size)
            running_results['batch_sizes'] += batch_size
    
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)
            print('fake_image', fake_img.size())
    
            netD.zero_grad()
            real_out = netD(real_img).mean()
            print('real_out', real_out)

            fake_out = netD(fake_img).mean()
            print('fake_out', fake_out)

            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()
    
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            
            
            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
    
        netG.eval()
        out_path = '/home-nfs2/local/VANDERBILT/kanakap/PycharmProjects/CDMRI-SRGAN/SRGAN/training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            outputs_sr = []
            for image_name, val_lr, val_hr, pad_idx, orig_shape in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)
                outputs_sr.append(sr.cpu())
        
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
                # print(val_hr_restore.size())
                # print(val_hr_restore.squeeze(0))
                # print(hr.data.size())
                # print(hr.data.cpu().squeeze(0))
                # print(sr.data.cpu().squeeze(0))
                #val_images.extend(
                    #[(val_hr_restore.unsqueeze(0)), (hr.data.cpu().unsqueeze(0)),
                     #(sr.data.cpu().unsqueeze(0))])

            #val_image = torch.stack(val_images
            #val_images = torch.chunk(val_images, val_images.size(0) // 15)
            #TODO SAVE IMAGES
            lx, lX, ly, lY, lz, lZ, rx, rX, ry, rY, rz, rZ = pad_idx
            #out = sr.data.cpu().squeeze()
            print(len(outputs_sr))
            out = torch.cat(outputs_sr, dim=0).squeeze()
            print('OUT_SHAPE', out.shape)
            out = out.permute(1, 2, 3, 0).detach().numpy()
            final_out = np.zeros(orig_shape)
            final_out[rx:rX, ry:rY, rz:rZ] = out[lx:lX, ly:lY, lz:lZ]
            ref = nib.load(image_name[0])
            nii = nib.Nifti1Image(final_out, affine=ref.affine, header=ref.header)
            print(out_path)
            nib.save(nii, out_path + 'image_result_%d.nii.gz' % (gen_image))
            gen_image += 1
            #val_save_bar = tqdm(val_images, desc='[saving training results]')
            #index = 1
            # for image in val_save_bar:
            #     #print(image)
            #     image = image.detach().cpu().numpy()
            #     #image = utils.make_grid(image, nrow=3, padding=5)
            #     #utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
            #     image.to_filename(out_path + 'epoch_%d_index_%d.nii.gz' % (epoch, index))
            #     index += 1
    
        # save model parameters
        #os.mkdir('/home-nfs2/local/VANDERBILT/kanakap/PycharmProjects/CDMRI-SRGAN/SRGAN/epochs')
        torch.save(netG.state_dict(), '/home/local/VANDERBILT/kanakap/PycharmProjects/CDMRI-SRGAN/SRGAN/epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), '/home/local/VANDERBILT/kanakap/PycharmProjects/CDMRI-SRGAN/SRGAN/epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    
        if epoch % 10 == 0 and epoch != 0:
            out_path = '/home/local/VANDERBILT/kanakap/PycharmProjects/CDMRI-SRGAN/SRGAN/statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
