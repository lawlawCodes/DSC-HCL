import torch
import torch.nn.functional as F
import numpy as np
import random  # 添加缺失的导入
import cv2


def generate_saliency_mask(img, lam=0.5):
    size = img.size()
    W = size[1]
    H = size[2]

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    temp_img = img.cpu().numpy().transpose(2, 1, 0)

    saliency = cv2.saliency.StaticSaliencyFineGrained.create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    if not success:
        x = np.random.randint(W)
        y = np.random.randint(H)
    else:
        threshMap = (saliencyMap * 255).astype("uint8")
        max_idx = np.argmax(threshMap)
        x, y = np.unravel_index(max_idx, threshMap.shape)

    x1 = np.clip(x - cut_w // 2, 0, W)
    y1 = np.clip(y - cut_h // 2, 0, H)

    x2 = np.clip(x + cut_w // 2, 0, W)
    y2 = np.clip(y + cut_h // 2, 0, H)

    # 创建全1的掩码 (H, W)
    mask = torch.ones((H, W))

    # 将显著区域置为0
    mask[y1:y2, x1:x2] = 0

    return mask.long()

def generate_cutmix_mask(img_size, ratio=2):
    cut_area = img_size[0] * img_size[1] / ratio
    cut_area = img_size[0] * img_size[1] / ratio
    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cut_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0

    return mask.long()

def generate_unsup_aug_ss(conf_w, mask_w, data_s):
    batch_size, _, im_h, im_w = data_s.shape
    device = data_s.device
    new_data_s, new_mask_w, new_conf_w = [], [], []
    for i in range(batch_size):
        augmix_mask = generate_saliency_mask(data_s[i]).to(device)
        new_data_s.append((data_s[i] * augmix_mask + data_s[(i + 1) % batch_size] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % batch_size] * (1 - augmix_mask)).unsqueeze(0))
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % batch_size] * (1 - augmix_mask)).unsqueeze(0))
    new_data_s, new_mask_w, new_conf_w = (torch.cat(new_data_s), torch.cat(new_mask_w), torch.cat(new_conf_w))

    return new_conf_w, new_mask_w, new_data_s

def generate_unsup_aug_ds(data_s1, data_s2):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_data_s = []
    for i in range(b):
        augmix_mask = generate_saliency_mask(data_s1[i]).to(device)
        new_data_s.append((data_s1[i] * augmix_mask + data_s2[i] * (1 - augmix_mask)).unsqueeze(0))
    new_data_s = torch.cat(new_data_s)

    return new_data_s

def generate_unsup_aug_dc(conf_w, mask_w, data_s1, data_s2):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_conf_w, new_mask_w, new_data_s = [], [], []
    for i in range(b):
        augmix_mask = generate_saliency_mask(data_s1[i]).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_data_s.append((data_s1[i] * augmix_mask + data_s2[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
    new_conf_w, new_mask_w, new_data_s = (torch.cat(new_conf_w), torch.cat(new_mask_w), torch.cat(new_data_s))

    return new_conf_w, new_mask_w, new_data_s

def generate_unsup_aug_sdc_1(conf_w, mask_w, data_s1, data_s2):
    batch_size, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_data_s, new_mask_w, new_conf_w = [], [], []
    for i in range(batch_size):
        if i % 2 == 0:
            augmix_mask = generate_saliency_mask(data_s1[i]).to(device)
            new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % batch_size] * (1 - augmix_mask)).unsqueeze(0))
            new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % batch_size] * (1 - augmix_mask)).unsqueeze(0))
            new_data_s.append(
                (data_s1[i] * augmix_mask + data_s2[(i + 1) % batch_size] * (1 - augmix_mask)).unsqueeze(0))
        else:
            augmix_mask = generate_saliency_mask(data_s2[i]).to(device)
            new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % batch_size] * (1 - augmix_mask)).unsqueeze(0))
            new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % batch_size] * (1 - augmix_mask)).unsqueeze(0))
            new_data_s.append(
                (data_s2[i] * augmix_mask + data_s1[(i + 1) % batch_size] * (1 - augmix_mask)).unsqueeze(0))

    new_data_s, new_mask_w, new_conf_w = (torch.cat(new_data_s), torch.cat(new_mask_w), torch.cat(new_conf_w))
    return new_conf_w, new_mask_w, new_data_s






