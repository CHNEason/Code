import torch
from torch.autograd import Variable as V
import cv2
import os
import numpy as np
from PIL import Image
import warnings
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from networks.dinknet import DinkNet34

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class CustomDataset(Dataset):
    def __init__(self, source_dir, gt_root):
        self.source_dir = source_dir
        self.gt_root = gt_root
        self.data = os.listdir(source_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        image_path = os.path.join(self.source_dir, img_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        label_formats = ['.png', '.tif', '_mask.png']
        ground_truth_path = None
        for label_format in label_formats:
            potential_label_path = os.path.join(self.gt_root, img_name.split('.')[0] + label_format)
            if os.path.exists(potential_label_path):
                ground_truth_path = potential_label_path
                break
            potential_label_path2 = os.path.join(self.gt_root, img_name.split('_')[0] + label_format)
            if os.path.exists(potential_label_path2):
                ground_truth_path = potential_label_path2
                break
        ground_truth = np.array(Image.open(ground_truth_path)).astype(np.uint8)
        if len(ground_truth.shape) == 3:
            ground_truth = ground_truth[:, :, 1]
        return {'image': image, 'ground_truth': ground_truth, 'filename': img_name}


def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    '''
    pred_mask = pred_mask.cpu().detach().numpy().astype(np.uint8)
    label = label.cpu().detach().numpy()
    TP = ((pred_mask == 1) & (label == 1)).sum()
    FN = ((pred_mask == 0) & (label == 1)).sum()
    TN = ((pred_mask == 0) & (label == 0)).sum()
    FP = ((pred_mask == 1) & (label == 0)).sum()
    acc = (TP + TN) / (TP + FN + TN + FP)
    sen = TP / (TP + FN)
    iou = TP / (TP + FN + FP)
    pre = TP / (TP + FP + 1e-6)
    f1 = (2 * pre * sen) / (pre + sen + 1e-6)
    return acc, sen, iou, pre, f1


class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()

    def test_one_img(self, img_tensor):
        self.net.eval()
        img_tensor = torch.permute(img_tensor, [0, 3, 1, 2])
        if tta_test == 1:
            img1 = torch.cat((img_tensor, torch.rot90(img_tensor, k=1, dims=[2, 3])), dim=0)
            img2 = torch.flip(img1, dims=[2])
            img3 = torch.flip(img1, dims=[3])
            img4 = torch.flip(img2, dims=[3])
            with torch.no_grad():
                mask1 = self.net(img1)
                mask2 = self.net(img2)
                mask3 = self.net(img3)
                mask4 = self.net(img4)
                mask5 = mask1 + torch.flip(mask2, dims=[2]) + torch.flip(mask3, dims=[3]) + torch.flip(torch.flip(mask4, dims=[3]), dims=[2])
                mask = mask5[:1,:,:,:] + torch.flip(torch.flip(torch.rot90(mask5[1:,:,:,:], k=1, dims=[2, 3]), dims=[2]), dims=[3])
            return mask.squeeze()/8
        else:
            with torch.no_grad():
                mask = self.net(img_tensor)
            return mask.squeeze()

    def load(self, path):
        state_dict = torch.load(path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        if isinstance(state_dict, tuple):
            state_dict = state_dict[0]
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)


tta_test = 1


def test_ce_net_vessel():
    source = r'C:\Users\13006\Desktop\Projects\Road Datasets\deep_6226\val'
    gt_root = r'C:\Users\13006\Desktop\Projects\Road Datasets\deep_6226\label'
    dataset = CustomDataset(source, gt_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    solver = TTAFrame(DinkNet34)
    solver.load(r'C:\Users\13006\Desktop\Projects\D-Link\weights\CycleGan(M-D)\First_Resi_G5_Skip-AdaN.pth')

    # target = r'C:\Users\13006\Desktop\Pre_Sty'
    # os.makedirs(target, exist_ok=True)

    total_acc = []
    total_sen = []
    total_iou = []
    total_pre = []
    total_f1 = []
    disc = 20
    # mylog = open(os.path.join(target, 'test_record.txt'), 'w', encoding='utf-8')

    threshold = 0.5

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        image = data['image'].cuda().float() / 255.0
        ground_truth = data['ground_truth'].cuda()
        filename = data['filename'][0]

        mask = solver.test_one_img(image)
        mask = ((mask > threshold) * 255).cpu().numpy().astype(np.uint8)

        ground_truth = ground_truth[0, :, :]
        if ground_truth.max() == 1:
            ground_truth *= 255

        predi_mask = np.zeros_like(mask)
        predi_mask[mask > disc] = 1

        acc, sen, iou, pre, f1 = accuracy(torch.tensor(predi_mask), torch.tensor(ground_truth/255))
        total_acc.append(acc)
        total_sen.append(sen)
        total_iou.append(iou)
        total_pre.append(pre)
        total_f1.append(f1)

        print(f'{i + 1} {filename} {iou:.4f}')
        # print(f'{i + 1} {filename} {iou:.4f}', file=mylog)
        # mylog.flush()

        # cv2.imwrite(os.path.join(target, filename.split('.')[0] + '.png'), mask)

    print(f'Mean accuracy: {np.mean(total_acc):.4f}, Std: {np.std(total_acc)}')
    print(f'Mean sensitivity: {np.mean(total_sen):.4f}, Std: {np.std(total_sen)}')
    print(f'Mean IoU: {np.mean(total_iou):.4f}, Std: {np.std(total_iou)}')
    print(f'Mean precision: {np.mean(total_pre):.4f}, Std: {np.std(total_pre)}')
    print(f'Mean F1-score: {np.mean(total_f1):.4f}, Std: {np.std(total_f1)}')
    print('---------------------')
    print('Mean IoU:', f'{np.mean(total_iou):.4f}, Std: {np.std(total_iou)}')

    # mylog.close()


if __name__ == '__main__':
    test_ce_net_vessel()
