"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import random

import torch
import torch.utils.data as data

import cv2
import numpy as np
import os

def random_crop(img, mask, crop_size):
    height, width = img.shape[:2]
    crop_h, crop_w = crop_size


    start_x = random.randint(0, width - crop_w)
    start_y = random.randint(0, height - crop_h)

    # 裁剪图像和掩码
    img_crop = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
    mask_crop = mask[start_y:start_y + crop_h, start_x:start_x + crop_w]

    return img_crop, mask_crop

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask


def default_loader(img_path, mask_path):

    img = cv2.imread(img_path)
    # print("img:{}".format(np.shape(img)))
    img = cv2.resize(img, (1024, 1024))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = 255. - cv2.resize(mask, (1024, 1024))
    
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
    
    mask = np.expand_dims(mask, axis=2)
    #
    # print(np.shape(img))
    # print(np.shape(masks))

    img = np.array(img, np.float32).transpose(2, 0, 1)/255.0 * 3.2 - 1.6

    return img, mask


def default_DRIVE_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 将标签输入改成跟论文一样的格式
    mask[mask > 0] = 255

    # 随机裁剪
    crop_size = (512, 512)
    img, mask = random_crop(img, mask, crop_size)

    # img = randomHueSaturationValue(img,
    #                                hue_shift_limit=(-30, 30),
    #                                sat_shift_limit=(-5, 5),
    #                                val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0

    return img, mask

def default_DRIVE_loader3(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 将标签输入改成跟论文一样的格式
    mask[mask > 0] = 255
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0

    return img, mask


def default_DRIVE_loader2(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 将标签输入改成跟论文一样的格式
    mask[mask > 0] = 255
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0

    return img, mask


def read_DRIVE_datasets(root_path, mode):
    images = []
    masks = []
    image_root = r'C:\Users\13006\Desktop\Projects\CycleGan\Models & Data\M-D\First_Dense_G5_Skip-AdaN\Trans'
    # image_root = r'C:\Users\13006\Desktop\Projects\CycleGan\Trans_FTL'
    # image_root = r'C:\Users\13006\Desktop\Projects\CycleGan\datasets\Road\test\Massa'
    # image_root2 = r'C:\Users\13006\Desktop\Projects\CycleGan\datasets\Road\test\Massa2'
    # image_root = r'C:\Users\13006\Desktop\Projects\Road Datasets\deep_6226\img'
    # image_root2 = r'C:\Users\13006\Desktop\Projects\CycleGan\datasets\Road2\test\DP2'
    # image_root3 = r'C:\Users\13006\Desktop\Projects\CycleGan\datasets\Road2\test\DP3'
    gt_root = r'C:\Users\13006\Desktop\Projects\Road Datasets\Ma\label'
    # gt_root = r'C:\Users\13006\Desktop\Projects\Road Datasets\deep_6226\label'
    # gt_root = r'C:\Users\13006\Desktop\Projects\Road Datasets\DP_1024'

    # lis1 = np.load(r'C:\Users\13006\Desktop\Projects\CycleGan\datasets\Road2\first_three.npz')['arr_0']
    # lis2 = np.load(r'C:\Users\13006\Desktop\Projects\CycleGan\datasets\Road2\second_three.npz')['arr_0']
    # lis3 = np.load(r'C:\Users\13006\Desktop\Projects\CycleGan\datasets\Road2\third_three.npz')['arr_0']

    # for image_name in lis1:
    #     image_path = os.path.join(image_root, image_name)
    #     image_path2 = os.path.join(image_root2, image_name)
    #     label_path = os.path.join(gt_root, image_name.split('_')[0] + '_mask.png')
    #     # label_path = os.path.join(gt_root, image_name.split('.')[0] + '.tif')
    #     images.append(image_path)
    #     images.append(image_path2)
    #     masks.append(label_path)
    #     masks.append(label_path)
    #
    # for image_name in lis2:
    #     image_path = os.path.join(image_root, image_name)
    #     image_path2 = os.path.join(image_root2, image_name)
    #     label_path = os.path.join(gt_root, image_name.split('_')[0] + '_mask.png')
    #     # label_path = os.path.join(gt_root, image_name.split('.')[0] + '.tif')
    #     images.append(image_path)
    #     images.append(image_path2)
    #     masks.append(label_path)
    #     masks.append(label_path)
    #
    # for image_name in lis3:
    #     image_path = os.path.join(image_root, image_name)
    #     image_path2 = os.path.join(image_root2, image_name)
    #     label_path = os.path.join(gt_root, image_name.split('_')[0] + '_mask.png')
    #     # label_path = os.path.join(gt_root, image_name.split('.')[0] + '.tif')
    #     images.append(image_path)
    #     images.append(image_path2)
    #     masks.append(label_path)
    #     masks.append(label_path)

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        # label_path = os.path.join(gt_root, image_name.split('_')[0] + '_mask.png')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '.tif')
        images.append(image_path)
        masks.append(label_path)

    # for image_name in os.listdir(image_root2):
    #     image_path = os.path.join(image_root2, image_name)
    #     # label_path = os.path.join(gt_root, image_name.split('_')[0] + '_mask.png')
    #     label_path = os.path.join(gt_root, image_name.split('.')[0] + '.tif')
    #     images.append(image_path)
    #     masks.append(label_path)

    return images, masks


def read_DRIVE_datasetsGD():
    images = []
    images2 = []
    masks = []
    image_root = r'C:\Users\asus\Desktop\Python\Road Datasets\Massa_6000(CYH)\img'
    image_root2 = r'C:\Users\asus\Desktop\Python\Dlink\dataset\M-D(histmatch)\val\img'
    gt_root = r'C:\Users\asus\Desktop\Python\Road Datasets\Massa_6000(CYH)\label'

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '.tif')
        images.append(image_path)
        masks.append(label_path)
    for image_name2 in os.listdir(image_root2):
        image_path2 = os.path.join(image_root2, image_name2)
        images2.append(image_path2)


    return images, images2, masks


def read_DRIVE_datasets2(root_path, mode):
    images = []
    masks = []
    # image_root = r'C:\Users\13006\Desktop\Projects\Road Datasets\Ma\test'
    # gt_root = r'C:\Users\13006\Desktop\Projects\Road Datasets\Ma\label'
    # image_root = r'C:\Users\13006\Desktop\Projects\Road Datasets\WHU_Test\img'
    # gt_root = r'C:\Users\13006\Desktop\Projects\Road Datasets\WHU_Test\label'
    image_root = r'C:\Users\13006\Desktop\Projects\Road Datasets\deep_6226\val'
    gt_root = r'C:\Users\13006\Desktop\Projects\Road Datasets\deep_6226\label'

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name.split('_')[0] + '_mask.png')
        # label_path = os.path.join(gt_root, image_name.split('.')[0] + '.tif')
        # label_path = os.path.join(gt_root, image_name.split('.')[0] + '.png')
        images.append(image_path)
        masks.append(label_path)

    return images, masks


def read_DRIVE_datasets3(root_path, mode):
    images = []
    masks = []
    image_root = r'C:\Users\asus\Desktop\Python\Dlink\dataset\ST\Dp_pseudo\img'
    # gt_root = r'C:\Users\asus\Desktop\Python\Dlink\dataset\ST\Dp_pseudo\label_G1+G2-2'
    gt_root = r'C:\Users\asus\Desktop\Python\Dlink\dataset\ST\Dp_pseudo\label_G1+G2-2'

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        # label_path = os.path.join(gt_root, image_name.split('_')[0] + '_mask.png')
        label_path = os.path.join(gt_root, image_name.split('_')[0] + '_sat.png')
        # label_path = os.path.join(gt_root, image_name.split('.')[0] + '.tif')
        images.append(image_path)
        masks.append(label_path)

    return images, masks

class TrainFolder(data.Dataset):
    def __init__(self, root_path, datasets='Road',  mode=None):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        if self.dataset == 'Road':
            self.images, self.labels = read_DRIVE_datasets(self.root, self.mode)

    def __getitem__(self, index):
        img, mask = default_DRIVE_loader(self.images[index], self.labels[index])
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)

class STFolder(data.Dataset):
    def __init__(self, root_path, datasets='Road',  mode=None):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        if self.dataset == 'Road':
            self.images, self.labels = read_DRIVE_datasets3(self.root, self.mode)

    def __getitem__(self, index):
        img, mask = default_DRIVE_loader3(self.images[index], self.labels[index])
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)

class ValFolder(data.Dataset):
    def __init__(self, root_path, datasets='Road',  mode=None):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        if self.dataset == 'Road':
            self.images, self.labels = read_DRIVE_datasets2(self.root, self.mode)

    def __getitem__(self, index):

        img, mask = default_DRIVE_loader2(self.images[index], self.labels[index])
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)
