from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.functional import resize, to_tensor, normalize
from PIL import Image
import os
import numpy as np
import random
import csv
import scipy.io
import xlrd


def default_loader(path):
    return Image.open(path).convert('RGB')  #


class IQADataset(Dataset):
    def __init__(self, args, status='training', loader=default_loader):

        self.status = status

        self.loader = loader
        self.args = args
        self.root = args.koniq_root

        imgname = []
        label=[]
        # print(status)

        imgname_ori = []
        mos_ori = []
        csv_file_ori = os.path.join(self.root, 'koniq10k_distributions_sets.csv')
        with open(csv_file_ori) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['set'] == status:
                    imgname_ori.append(row['image_name'])
                    mos = np.array(float(row['MOS'])).astype(np.float32)
                    mos_ori.append(mos)

        csv_file = os.path.join(self.root, 'koniq10k_extend.csv')
        # csv_file = os.path.join(self.root, 'koniq++database.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    dist_label_temp = np.zeros(5)
                    mos = mos_ori[imgname_ori.index(row['filename'])]
                    dist_label_temp[0] = np.array(float(mos)).astype(np.float32)
                    dist_label_temp[1] = np.array(float(row['artifacts'])).astype(np.float32) * 100
                    dist_label_temp[2] = np.array(float(row['blur'])).astype(np.float32) * 100
                    dist_label_temp[3] = np.array(float(row['contrast'])).astype(np.float32) * 100
                    dist_label_temp[4] = np.array(float(row['colors'])).astype(np.float32) * 100
                    label.append(dist_label_temp)
                    imgname.append(row['filename'])
                except:
                    continue
        label = np.array(label).astype(np.float32)

        self.label = label
        self.label_std = label
        self.im_names = imgname
        self.ims = []

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im_temp = self.loader(os.path.join(self.root, '1024x768', self.im_names[idx]))
        im = (resize(im_temp, (self.args.resize_size_h, self.args.resize_size_w)))
        im = transform(im)

        label = self.label[idx]
        label_std = self.label_std[idx]

        return im, (label, label_std)


class IQADataset_LIVEC(Dataset):
    def __init__(self, args, status='training', loader=default_loader):

        self.status = status

        self.loader = loader
        self.args = args
        self.root = args.clive_root

        imgpath = scipy.io.loadmat(os.path.join(args.clive_root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:1169]
        mos = scipy.io.loadmat(os.path.join(args.clive_root, 'Data', 'AllMOS_release.mat'))
        labels = mos['AllMOS_release'].astype(np.float32)
        labels = labels[0][7:1169]

        sample = []
        for i, item in enumerate(imgpath):
            sample.append(os.path.join(args.clive_root, 'Images', item[0][0]))

        label = np.array(labels).astype(np.float32)

        self.label = label
        self.label_std = label
        self.im_names = sample
        self.ims = []

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im_temp = self.loader(os.path.join(self.root, 'Images', self.im_names[idx]))
        im_temp = (resize(im_temp, (self.args.resize_size_h, self.args.resize_size_w)))
        im= transform(im_temp)

        label = self.label[idx]
        label_std = self.label_std[idx]

        return im, (label, label_std)


class IQADataset_SPAQ(Dataset):
    def __init__(self, args, status='training', loader=default_loader):

        self.status = status

        self.loader = loader
        self.args = args
        self.root = args.spaq_root

        work_book = xlrd.open_workbook(os.path.join(args.spaq_root, 'MOS and Image attribute scores.xlsx'))

        sheet_1 = work_book.sheet_by_index(0)

        imgname = []
        label = []

        for row in range(1, sheet_1.nrows):
            imgname.append(sheet_1.row_values(row)[0])
            label.append(sheet_1.row_values(row)[1:])

        label = np.array(label).astype(np.float32)

        self.label = label
        self.label_std = label
        self.im_names = imgname

        self.ims = []

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im_temp = self.loader(os.path.join(self.root, 'TestImage', self.im_names[idx]))
        im = (resize(im_temp, (self.args.resize_size_h, self.args.resize_size_w)))
        im = transform(im)

        label = np.array([self.label[idx][0], self.label[idx][4], self.label[idx][5], self.label[idx][3], self.label[idx][2]])
        label_std = label

        return im, (label, label_std)


def transform(im):
    im = to_tensor(im)
    im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return im


def get_data_loaders(args):
    """ Prepare the train-val-test data
    :param args: related arguments
    :return: train_loader, val_loader, test_loader
    """
    train_dataset = IQADataset(args, 'training')
    batch_size = args.batch_size
    if args.debug:
        num_samples = 5 * batch_size
        print("Debug mode: reduced training dataset to the first {} samples".format(num_samples))
        train_dataset = Subset(train_dataset, list(range(num_samples)))

    else:
        train_dataset = Subset(train_dataset, list(range(7058)))
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True)  # If the last batch only contains 1 sample, you need drop_last=True.
    val_dataset = IQADataset(args, 'validation')
    test_dataset = IQADataset(args, 'test')
    if args.debug:
        num_samples = 5
        print("Debug mode: reduced validation/test datasets to the first {} samples".format(num_samples))
        val_dataset = Subset(val_dataset, list(range(num_samples)))
        test_dataset = Subset(test_dataset, list(range(num_samples)))

    else:
        val_dataset = Subset(val_dataset, list(range(1000)))
        test_dataset = Subset(test_dataset, list(range(2015)))

    val_loader = DataLoader(val_dataset)    
    test_loader = DataLoader(test_dataset)
    return train_loader, val_loader, test_loader
