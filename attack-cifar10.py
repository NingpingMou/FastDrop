# code for cifar 10
import glob
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import scipy
import os
from typing import List
import numpy as np
import cv2

import torch
import torchvision.transforms as transforms
import torchvision.models as models

from models import ResNet18
from utils import UnNorm, load_image_uint8

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print("got GPU")

log_file = 'cifar10.txt'
img_path = "data/cifar10"
if not os.path.exists(img_path+'_adv'):
    os.mkdir(img_path+'_adv')


square_max_num = 32
l2_norm_thres = 5.0

norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
trans1 = transforms.ToTensor()
un_norm = UnNorm(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))


def square_avg(freq:np.ndarray, index:int):
    rank1 = np.sum(freq[index, index:square_max_num-index, :])
    rank2 = np.sum(freq[square_max_num-1-index, index:square_max_num-index, :])
    col1 = np.sum(freq[index+1:square_max_num-1-index, index, :])
    col2 = np.sum(freq[index+1:square_max_num-1-index, square_max_num-1-index, :])
    num = 4*(square_max_num - 2*index) - 2

    return (rank1+rank2+col1+col2) / float(num)

def square_zero(freq:np.ndarray, index:int):
    freq_modified = freq.copy()
    freq_modified[index, index:square_max_num-index, :] = 0
    freq_modified[square_max_num-1-index, index:square_max_num-index, :] = 0
    freq_modified[index:square_max_num-index:, index, :] = 0
    freq_modified[index:square_max_num-index, square_max_num-1-index, :] = 0

    return freq_modified

def square_recover(freq_modified:np.ndarray, freq_ori:np.ndarray, index:int):
    freq_modified[index, index:square_max_num-index, :] = freq_ori[index, index:square_max_num-index, :]
    freq_modified[square_max_num-1-index, index:square_max_num-index, :] = freq_ori[square_max_num-1-index, index:square_max_num-index, :]
    freq_modified[index:square_max_num-index:, index, :] = freq_ori[index:square_max_num-index:, index, :]
    freq_modified[index:square_max_num-index, square_max_num-1-index, :] = freq_ori[index:square_max_num-index, square_max_num-1-index, :]

    return freq_modified


def freq_to_np_img(freq:np.ndarray):
    img = np.abs(np.fft.ifft2(freq, axes=(0, 1)))
    img = np.clip(img, 0, 255)  
    img = img.astype('uint8')

    return img


def freq_to_img(freq:np.ndarray):
    img = np.abs(np.fft.ifft2(freq, axes=(0, 1)))
    img = np.clip(img, 0, 255) 
    img = img.astype('uint8')
    img = trans1(img).unsqueeze(0)
    img = norm(img)

    return img


def cal_norm_of_freq(freq1:np.ndarray, freq2:np.ndarray, norm:str='l2'):
    img1 = freq_to_img(freq1)
    img1 = img1.to(device)
    img2 = freq_to_img(freq2)
    img2 = img2.to(device)

    if norm == 'l2':
        return torch.norm(un_norm(img1.squeeze()) - un_norm(img2.squeeze()), p=2).item()
    else:
        return torch.norm(un_norm(img1.squeeze()) - un_norm(img2.squeeze()), p=np.inf).item()


def ifft_predict(net:torch.nn.Module, freq:np.ndarray, device=torch.device("cuda:0")):
    img_adv = freq_to_img(freq)
    img_adv = img_adv.to(device)
    with torch.no_grad():
        out = net(img_adv)
        _, adv_label = torch.max(out, dim=1)

    return adv_label





def fastdrop(net:torch.nn.Module, img:np.ndarray, block_size:int=16, file_path:str='', device=torch.device("cuda:0")):
    path1, path2 = os.path.split(file_path)
    save_path = path1 + '_adv/' + path2
    query_num = 0
    ori_img = img.copy()

    img = trans1(img).unsqueeze(0)
    img = norm(img)
    img = img.to(device)
    query_num += 1
    with torch.no_grad():
        out = net(img)
        _, ori_label = torch.max(out, dim=1)

    # DFT
    # fft to original numpy image
    freq = np.fft.fft2(ori_img, axes=(0, 1))
    freq_ori = freq.copy()
    freq_ori_m = np.abs(freq_ori)
    freq_abs = np.abs(freq)
    num_block = int(square_max_num/2)
    block_sum = np.zeros(num_block)
    for i in range(num_block):
            block_sum[i] = square_avg(freq_abs, i)

    # ordered index
    block_sum_ind = np.argsort(block_sum)
    # block_sum_ind = block_sum_ind[::-1]
    block_sum_ind_flag = np.zeros(num_block)

    with open(log_file, 'a') as f:
        print('second stage!!!', file=f)
    img_save = None
    range_0 = [1,2]
    range_1 = range(3, num_block+1, 1)
    mags = range_0 + list(range_1)
    freq_sec_stage = freq.copy()
    freq_sec_stage_m = np.abs(freq_sec_stage)  
    freq_sec_stage_p = np.angle(freq_sec_stage)  
    mag_start = 0
    for mag in mags:
        for i in range(mag_start,mag):
            ind = block_sum_ind[i]
            freq_sec_stage_m = square_zero(freq_sec_stage_m, ind)
            freq_sec_stage = freq_sec_stage_m * np.e ** (1j * freq_sec_stage_p)

        img_adv = np.abs(np.fft.ifft2(freq_sec_stage, axes=(0, 1)))
        img_adv = np.clip(img_adv, 0, 255)  
        img_adv = img_adv.astype('uint8')
        img_save = img_adv.copy()
        img_adv = trans1(img_adv).unsqueeze(0)
        img_adv = norm(img_adv)
        img_adv = img_adv.to(device)
        query_num += 1
        with torch.no_grad():
            out = net(img_adv)
            _, adv_label = torch.max(out, dim=1)

        mag_start = mag
        if ori_label != adv_label:
            print('max num success')
            print('%d block' % (mag))
            print('l2_norm: ', torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=2).item())
            print('linf_norm: ', torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=np.inf).item())
            l2_norm = torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=2)
            if l2_norm.item() < l2_norm_thres:
                with open(log_file, 'a') as f:
                    print('%d block success' % (mag), file=f)
                    
                img_save = Image.fromarray(img_save)
                img_save.save(save_path)
                with open(log_file, 'a') as f:
                    print('query number: ', query_num, file=f)
                    print('l2_norm: ', torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=2).item(), file=f)
                    print('linf_norm: ', torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=np.inf).item(), file=f)
                return
            else:
                with open(log_file, 'a') as f:
                    print('success adv: ', mag_start, file=f)
                    print('l2_norm: ', torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=2).item(), file=f)
                    print('linf_norm: ', torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=np.inf).item(), file=f)

            break


    # get adv example
    with open(log_file, 'a') as f:
        print('third stage!!!', file=f)
    # img_save = None
    img_temp = img_save
    # max_i = -1
    max_i = mag_start - 1
    block_sum_ind_flag[:max_i+1] = 1
    print('max_i: ', max_i)
    # freq_m = np.abs(freq)
    freq_m = freq_sec_stage_m
    freq_p = np.angle(freq)
    

    # optimize the adv example
    optimize_block = 0
    l2_norm = torch.tensor(0)
    linf_norm = torch.tensor(0)
    with open(log_file, 'a') as f:
        print('fourth stage!!!', file=f)
    for round in range(2):
        with open(log_file, 'a') as f:
            print('round: ', round, file=f)
        for i in range(max_i, -1, -1):
            if block_sum_ind_flag[i] == 1:
                ind = block_sum_ind[i]
                freq_m = square_recover(freq_m, freq_ori_m, ind)
                freq = freq_m * np.e ** (1j * freq_p)

                img_adv = np.abs(np.fft.ifft2(freq, axes=(0, 1)))
                img_adv = np.clip(img_adv, 0, 255)  # 会产生一些过大值需要截断
                img_adv = img_adv.astype('uint8')
                img_temp_2 = img_adv.copy()
                img_adv = trans1(img_adv).unsqueeze(0)
                img_adv = norm(img_adv)
                img_adv = img_adv.to(device)
                query_num += 1
                with torch.no_grad():
                    out = net(img_adv)
                    _, adv_label = torch.max(out, dim=1)

                if adv_label == ori_label:
                    freq_m = square_zero(freq_m, ind)
                    freq = freq_m * np.e ** (1j * freq_p)
                    # freq = square_zero(freq, ind)   # accident，这是错误的
                else:
                    img_temp = img_temp_2.copy()
                    optimize_block += 1
                    # print('optimize block: ', i)
                    l2_norm = torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=2)
                    linf_norm = torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=np.inf)
                    # print(l2_norm.item())
                    block_sum_ind_flag[i] = 0
        if optimize_block == 0:
            l2_norm = torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=2)
            linf_norm = torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=np.inf)
        with open(log_file, 'a') as f:
            print('optimize block number: ', optimize_block, file=f)
            print('l2_norm: ', l2_norm.item(), file=f)
            print('linf_norm: ', linf_norm.item(), file=f)

    with open(log_file, 'a') as f:
        print('final result', file=f)
        print('original_que number: ', query_num, file=f)
        print('optimize block number: ', optimize_block, file=f)
        print('zero block number: ', np.sum(block_sum_ind_flag), file=f)


    img_temp = Image.fromarray(img_temp)
    img_temp.save(save_path)
    with open(log_file, 'a') as f:
        print('query number: ', query_num, file=f)
        print('l2_norm: ', l2_norm.item(), file=f)
        print('linf_norm: ', linf_norm.item(), file=f)



# load model
net = ResNet18().to(device)
net.load_state_dict(torch.load('ResNet10-CIFAR10.pth',map_location = device)['net'])
net.eval()



images = sorted(glob.glob(img_path + "/*png"))
num = 0
for file_path in images:
    num += 1
    if num == 1001:
        break
    print('\nimage number: ', num)
    with open(log_file, 'a') as f:
        print('\nimage number: ', num, file=f)
    img = load_image_uint8(data_name='CIFAR10', data_path=file_path)
    fastdrop(net, img, 16, file_path, device)