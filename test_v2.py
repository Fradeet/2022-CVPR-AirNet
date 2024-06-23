import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset, SRDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor

from net.model_v2 import AirNet_v2 as AirNet


def test_Denoise(net, dataset, sigma=15):
    output_path = opt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(x_query=degrad_patch, x_key=degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Deonise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))


def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = opt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(x_query=degrad_patch, x_key=degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))

def test_SR(net, dataset, task="Set5", scale:int=2):
    '''
    Test SR task, each dataset image is tested.
    net: PyTorch model,
    dataset: PyTorch Dataset, first is lq and second is gt,
    task: str, just task name,
    scale: int, scale factor.
    '''
    output_path = opt.output_path + task + '/x' + str(scale) + '/'
    # subprocess.check_output(['mkdir', '-p', output_path])  # Disable on Windows

    # dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(x_query=degrad_patch, x_key=degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one, 4 for SR')

    parser.add_argument('--denoise_path', type=str, default="test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="test/dehaze/", help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/", help='checkpoint save path')
    
    parser.add_argument('--SR_path', type=str, default="test/SR/", help='save path of test SR images')
    parser.add_argument('--scale', type=int, default=2, help='SR scales.')
    parser.add_argument('--num_feats', type=int, default=64, help='')
    opt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(opt.cuda)

    if opt.mode == 0:
        opt.batch_size = 3
        ckpt_path = opt.ckpt_path + 'Denoise.pth'
    elif opt.mode == 1:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'Derain.pth'
    elif opt.mode == 2:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'Dehaze.pth'
    elif opt.mode == 3:
        opt.batch_size = 5
        ckpt_path = opt.ckpt_path + 'All.pth'
    elif opt.mode == 4:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'SR.pth'

    denoise_set = DenoiseTestDataset(opt)
    derain_set = DerainDehazeDataset(opt)
    
    sr_set5_set = SRDataset(opt, scale=opt.scale, task="Set5")
    sr_Set14_set = SRDataset(opt, scale=opt.scale, task="Set14")
    
    # Make network
    net = AirNet(opt).cuda()
    net.eval()
    net.load_state_dict(torch.load(ckpt_path, map_location=torch.device(opt.cuda)))

    if opt.mode == 0:
        print('Start testing Sigma=15...')
        test_Denoise(net, denoise_set, sigma=15)

        print('Start testing Sigma=25...')
        test_Denoise(net, denoise_set, sigma=25)

        print('Start testing Sigma=50...')
        test_Denoise(net, denoise_set, sigma=50)
    elif opt.mode == 1:
        print('Start testing rain streak removal...')
        test_Derain_Dehaze(net, derain_set, task="Rain100L")
    elif opt.mode == 2:
        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="SOTS_outdoor")
    elif opt.mode == 3:
        print('Start testing Sigma=15...')
        test_Denoise(net, denoise_set, sigma=15)

        print('Start testing Sigma=25...')
        test_Denoise(net, denoise_set, sigma=25)

        print('Start testing Sigma=50...')
        test_Denoise(net, denoise_set, sigma=50)

        print('Start testing rain streak removal...')
        test_Derain_Dehaze(net, derain_set, task="derain")

        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze")
    elif opt.mode == 4:
        print('Start testing Set5...')
        test_SR(net, sr_set5_set, task="Set5")
        
        print('Start testing Set14...')
                
