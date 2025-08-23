import os
import torch
import pickle
import datetime
import logging
import random
import numpy as np
import torch.nn as nn
import PIL.Image as Image
import torch.nn.functional as F
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.utils.data import Dataset
from util.util_tools import MyCommon
from torch.utils.data import DataLoader
import tifffile


class Evaluator:
    ignore_index = 255

    @classmethod
    def classify_prediction(cls, pred_label, batch):
        gt_label = batch.get('query_label')

        # Apply ignore_index in PASCAL-5i labels (following evaluation scheme in PFE-Net (TPAMI 2020))
        query_ignore_idx = batch.get('query_ignore_idx')
        if query_ignore_idx is not None:
            assert torch.logical_and(query_ignore_idx, gt_label).sum() == 0
            query_ignore_idx *= cls.ignore_index
            gt_label = gt_label + query_ignore_idx
            pred_label[gt_label >= cls.ignore_index] = cls.ignore_index

        # compute intersection and union of each episode in a batch
        area_inter, area_pred, area_gt = [],  [], []
        for _pred_label, _gt_label in zip(pred_label, gt_label):
            _inter = _pred_label[_pred_label == _gt_label]
            if _inter.size(0) == 0:  # as torch.histc returns error if it gets empty tensor (pytorch 1.5.1)
                _area_inter = torch.tensor([0, 0], device=_pred_label.device)
            else:
                _area_inter = torch.histc(_inter, bins=2, min=0, max=1)
            area_inter.append(_area_inter)
            area_pred.append(torch.histc(_pred_label, bins=2, min=0, max=1))
            area_gt.append(torch.histc(_gt_label, bins=2, min=0, max=1))
        area_inter = torch.stack(area_inter).t()
        area_pred = torch.stack(area_pred).t()
        area_gt = torch.stack(area_gt).t()
        area_union = area_pred + area_gt - area_inter

        return area_inter, area_union

    pass


class DatasetISAID(Dataset):

    def __init__(self, datapath, fold, transform, split, shot, use_mask=False, mask_num=128):
        self.split = 'val' if split in ['val', 'test'] else 'train'
        self.fold = fold
        self.nfolds = 3
        self.nclass = 15
        self.benchmark = 'isaid'
        self.shot = shot
        self.datapath = datapath
        self.use_mask = use_mask
        self.mask_num = mask_num

        self.img_path = os.path.join(datapath, self.split, 'images')
        self.ann_path = os.path.join(datapath, self.split, 'semantic_png')
        self.pkl_path = os.path.join(datapath, self.split, 'sam_mask_vit_h_t64_p32_s50')

        self.transform = transform
        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        pass

    def get_data_and_mask(self, pkl_name):
        pkl = os.path.join(self.pkl_path, pkl_name) + '.pkl'
        with open(pkl, "rb") as f:
            image_label_mask = pickle.load(f)

        query_img = Image.fromarray(image_label_mask["image"])
        query_label = Image.fromarray(image_label_mask["label"])
        query_mask = np.asarray([MyCommon.rle_to_mask(one)
                                 for one in image_label_mask["masks"]], dtype=np.int8)
        return query_img, query_label, query_mask

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_label, query_mask, support_imgs, support_labels_c, \
        support_masks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_label = torch.tensor(np.array(query_label))
        query_mask = torch.tensor(np.array(query_mask))

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        support_labels_c = [torch.tensor(np.array(one)) for one in support_labels_c]
        support_masks = [torch.tensor(np.array(one)) for one in support_masks]
        support_masks = torch.stack(support_masks)

        query_label = F.interpolate(query_label.unsqueeze(0).unsqueeze(0).float(),
                                    query_img.size()[-2:], mode='nearest').squeeze()
        query_label, query_ignore_idx = self.extract_ignore_idx(query_label.float(), class_sample)

        support_labels, support_ignore_idxs = [], []
        for slabel in support_labels_c:
            slabel = F.interpolate(slabel.unsqueeze(0).unsqueeze(0).float(),
                                   support_imgs.size()[-2:], mode='nearest').squeeze()
            support_label, support_ignore_idx = self.extract_ignore_idx(slabel, class_sample)
            support_labels.append(support_label)
            support_ignore_idxs.append(support_ignore_idx)
        support_labels = torch.stack(support_labels)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_label': query_label,
                 'query_mask': query_mask[:self.mask_num],
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_labels': support_labels,
                 'support_masks': support_masks[:,:self.mask_num],
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'class_id': torch.tensor(class_sample)}
        return batch

    def extract_ignore_idx(self, label, class_id):
        boundary = (label / 255).floor()
        label[label != class_id + 1] = 0
        label[label == class_id + 1] = 1

        return label, boundary

    def load_frame(self, query_name, support_names):
        if self.use_mask:
            query_img, query_label, query_mask = self.get_data_and_mask(query_name)
            _support_data = [self.get_data_and_mask(name) for name in support_names]
            support_imgs = [one[0] for one in _support_data]
            support_labels = [one[1] for one in _support_data]
            support_masks = [one[2] for one in _support_data]
        else:
            query_img = self.read_img(query_name)
            query_label = self.read_label(query_name)
            query_mask = self.read_label(query_name)
            support_imgs = [self.read_img(name) for name in support_names]
            support_labels = [self.read_label(name) for name in support_names]
            support_masks = [self.read_label(name) for name in support_names]
            pass
        return query_img, query_label, query_mask, \
               support_imgs, support_labels, support_masks, query_img.size

    def read_label(self, img_name):
        return Image.open(os.path.join(self.ann_path, img_name) + '_instance_color_RGB.png')

    def read_img(self, img_name):
        return Image.open(os.path.join(self.img_path, img_name) + '.png')

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        nclass_train = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_train + i for i in range(nclass_train)]
        class_ids_train = [x for x in range(self.nclass) if x not in class_ids_val]
        return class_ids_train if self.split == 'train' else class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            split = "trn" if split == "train" else split
            fold_n_metadata = os.path.join('data/splits/isaid/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'train':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)
        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise

    pass


class DatasetDLRSD(Dataset):

    def __init__(self, datapath, fold, transform, split, shot, use_mask=False, mask_num=128):
        self.split = 'val' if split in ['val', 'test'] else 'train'
        self.fold = fold
        self.nfolds = 3
        self.nclass = 15
        self.benchmark = 'dlrsd'
        self.shot = shot
        self.datapath = datapath
        self.use_mask = use_mask # 是否使用预提取的 mask（如 SAM
        self.mask_num = mask_num # 使用的最大 mask 数量

        self.img_path = os.path.join(datapath, 'Images')
        self.ann_path = os.path.join(datapath, 'Labels')
        self.pkl_path = os.path.join(datapath, 'sam_mask_vit_h_t64_32_s50')

        self.transform = transform
        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        pass

    def get_data_and_mask(self, pkl_name):
        pkl = os.path.join(self.pkl_path, pkl_name) + '.pkl'
        with open(pkl, "rb") as f:
            image_label_mask = pickle.load(f)

        query_img = Image.fromarray(image_label_mask["image"])
        query_label = Image.fromarray(image_label_mask["label"])
        query_mask = np.asarray([MyCommon.rle_to_mask(one)
                                 for one in image_label_mask["masks"]], dtype=np.int8)
        return query_img, query_label, query_mask

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_label, query_mask, support_imgs, support_labels_c, \
        support_masks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_label = torch.tensor(np.array(query_label))
        query_mask = torch.tensor(np.array(query_mask))

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        support_labels_c = [torch.tensor(np.array(one)) for one in support_labels_c]
        support_masks = [torch.tensor(np.array(one)) for one in support_masks]
        support_masks = torch.stack(support_masks)

        query_label = F.interpolate(query_label.unsqueeze(0).unsqueeze(0).float(),
                                    query_img.size()[-2:], mode='nearest').squeeze()
        query_label, query_ignore_idx = self.extract_ignore_idx(query_label.float(), class_sample)

        support_labels, support_ignore_idxs = [], []
        for slabel in support_labels_c:
            slabel = F.interpolate(slabel.unsqueeze(0).unsqueeze(0).float(),
                                   support_imgs.size()[-2:], mode='nearest').squeeze()
            support_label, support_ignore_idx = self.extract_ignore_idx(slabel, class_sample)
            support_labels.append(support_label)
            support_ignore_idxs.append(support_ignore_idx)
        support_labels = torch.stack(support_labels)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_label': query_label,
                 'query_mask': query_mask[:self.mask_num],
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_labels': support_labels,
                 'support_masks': support_masks[:,:self.mask_num],
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'class_id': torch.tensor(class_sample)}
        return batch

    def extract_ignore_idx(self, label, class_id):
        # label_np = np.array(label)
        # print(f"[DEBUG] class_id: {class_id}, expected pixel: {class_id + 1}")
        # print(f"[DEBUG] label unique pixel values: {np.unique(label_np)}")
        boundary = (label / 255).floor()
        label[label != class_id + 1] = 0
        label[label == class_id + 1] = 1

        return label, boundary

    def load_frame(self, query_name, support_names):
        if self.use_mask:
            query_img, query_label, query_mask = self.get_data_and_mask(query_name)
            _support_data = [self.get_data_and_mask(name) for name in support_names]
            support_imgs = [one[0] for one in _support_data]
            support_labels = [one[1] for one in _support_data]
            support_masks = [one[2] for one in _support_data]
        else:
            query_img = self.read_img(query_name)
            query_label = self.read_label(query_name)
            query_mask = self.read_label(query_name)
            support_imgs = [self.read_img(name) for name in support_names]
            support_labels = [self.read_label(name) for name in support_names]
            support_masks = [self.read_label(name) for name in support_names]
            pass
        return query_img, query_label, query_mask, \
               support_imgs, support_labels, support_masks, query_img.size

    def read_label(self, img_name):
        path = os.path.join(self.ann_path, img_name[:-2], img_name) + '.png'
        label_img = Image.open(path)
        label_np = np.array(label_img)  # 保留调色板索引作为类别
        return Image.fromarray(label_np.astype(np.uint8))  # 转成 Image，保持一致性



    def read_img(self, img_name):
        return Image.open(os.path.join(self.img_path, img_name[:-2], img_name) + '.tif')

    def sample_episode(self, idx):
        if self.split == "train":
            class_sample = self.class_ids[idx % len(self.class_ids)]
            query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        else:
            query_name, class_sample = self.img_metadata[idx]
            pass

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break
            pass
        return query_name, support_names, class_sample

    def build_class_ids(self):
        nclass_train = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_train + i for i in range(nclass_train)]
        class_ids_train = [x for x in range(self.nclass) if x not in class_ids_val]
        return class_ids_train if self.split == 'train' else class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            split = "trn" if split == "train" else split
            fold_n_metadata = os.path.join('data/splits/DLRSD/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'train':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)
        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise

    pass

class DatasetLCMA(Dataset):

    def __init__(self, datapath, fold, transform, split, shot, use_mask=False, mask_num=128):
        self.split = 'val' if split in ['val', 'test'] else 'train'
        self.fold = fold
        self.nfolds = 3
        self.nclass = 18
        self.benchmark = 'lcma'
        self.shot = shot
        self.datapath = datapath
        self.use_mask = use_mask # 是否使用预提取的 mask（如 SAM
        self.mask_num = mask_num # 使用的最大 mask 数量

        self.img_path = os.path.join(datapath, 'Images_cls')
        self.ann_path = os.path.join(datapath, 'Labels_cls')
        self.pkl_path = os.path.join(datapath, 'sam_mask_vit_h_t64_32_s50')

        self.transform = transform
        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        pass

    def get_data_and_mask(self, pkl_name):
        pkl = os.path.join(self.pkl_path, pkl_name) + '.pkl'
        with open(pkl, "rb") as f:
            image_label_mask = pickle.load(f)

        query_img = Image.fromarray(image_label_mask["image"])
        query_label = Image.fromarray(image_label_mask["label"])
        query_mask = np.asarray([MyCommon.rle_to_mask(one)
                                 for one in image_label_mask["masks"]], dtype=np.int8)
        return query_img, query_label, query_mask

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_label, query_mask, support_imgs, support_labels_c, \
        support_masks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_label = torch.tensor(np.array(query_label))
        query_mask = torch.tensor(np.array(query_mask))

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        support_labels_c = [torch.tensor(np.array(one)) for one in support_labels_c]
        support_masks = [torch.tensor(np.array(one)) for one in support_masks]
        support_masks = torch.stack(support_masks)

        query_label = F.interpolate(query_label.unsqueeze(0).unsqueeze(0).float(),
                                    query_img.size()[-2:], mode='nearest').squeeze()
        query_label, query_ignore_idx = self.extract_ignore_idx(query_label.float(), class_sample)

        support_labels, support_ignore_idxs = [], []
        for slabel in support_labels_c:
            slabel = F.interpolate(slabel.unsqueeze(0).unsqueeze(0).float(),
                                   support_imgs.size()[-2:], mode='nearest').squeeze()
            support_label, support_ignore_idx = self.extract_ignore_idx(slabel, class_sample)
            support_labels.append(support_label)
            support_ignore_idxs.append(support_ignore_idx)
        support_labels = torch.stack(support_labels)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_label': query_label,
                 'query_mask': query_mask[:self.mask_num],
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_labels': support_labels,
                 'support_masks': support_masks[:,:self.mask_num],
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'class_id': torch.tensor(class_sample)}
        return batch

    def extract_ignore_idx(self, label, class_id):
        # label_np = np.array(label)
        # print(f"[DEBUG] class_id: {class_id}, expected pixel: {class_id + 1}")
        # print(f"[DEBUG] label unique pixel values: {np.unique(label_np)}")
        boundary = (label / 255).floor()
        label[label != class_id + 1] = 0
        label[label == class_id + 1] = 1

        return label, boundary

    def load_frame(self, query_name, support_names):
        if self.use_mask:
            query_img, query_label, query_mask = self.get_data_and_mask(query_name)
            _support_data = [self.get_data_and_mask(name) for name in support_names]
            support_imgs = [one[0] for one in _support_data]
            support_labels = [one[1] for one in _support_data]
            support_masks = [one[2] for one in _support_data]
        else:
            query_img = self.read_img(query_name)
            query_label = self.read_label(query_name)
            query_mask = self.read_label(query_name)
            support_imgs = [self.read_img(name) for name in support_names]
            support_labels = [self.read_label(name) for name in support_names]
            support_masks = [self.read_label(name) for name in support_names]
            pass
        return query_img, query_label, query_mask, \
               support_imgs, support_labels, support_masks, query_img.size

    def read_label(self, img_name):
        path = os.path.join(self.ann_path, img_name[:-2], img_name) + '.png'
        label_img = Image.open(path)
        label_np = np.array(label_img)  # 保留调色板索引作为类别
        return Image.fromarray(label_np.astype(np.uint8))  # 转成 Image，保持一致性



    def read_img(self, img_name):
        return Image.open(os.path.join(self.img_path, img_name[:-2], img_name) + '.tif')

    def sample_episode(self, idx):
        if self.split == "train":
            class_sample = self.class_ids[idx % len(self.class_ids)]
            query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        else:
            query_name, class_sample = self.img_metadata[idx]
            pass

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break
            pass
        return query_name, support_names, class_sample

    def build_class_ids(self):
        nclass_train = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_train + i for i in range(nclass_train)]
        class_ids_train = [x for x in range(self.nclass) if x not in class_ids_val]
        return class_ids_train if self.split == 'train' else class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            split = "trn" if split == "train" else split
            fold_n_metadata = os.path.join('data/splits/DLRSD/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'train':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)
        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise

    pass
class DatasetLCML(Dataset):

    def __init__(self, datapath, fold, transform, split, shot, use_mask=False, mask_num=128):
        self.split = 'val' if split in ['val', 'test'] else 'train'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 16
        self.benchmark = 'lcml'
        self.shot = shot
        self.datapath = datapath
        self.use_mask = use_mask # 是否使用预提取的 mask（如 SAM
        self.mask_num = mask_num # 使用的最大 mask 数量

        self.img_path = os.path.join(datapath, 'Images')
        self.ann_path = os.path.join(datapath, 'Labels')
        self.pkl_path = os.path.join(datapath, 'sam_mask_vit_h_t64_32_s50')

        self.transform = transform
        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        pass

    def get_data_and_mask(self, pkl_name):
        pkl = os.path.join(self.pkl_path, pkl_name) + '.pkl'
        with open(pkl, "rb") as f:
            image_label_mask = pickle.load(f)

        query_img = Image.fromarray(image_label_mask["image"])
        query_label = Image.fromarray(image_label_mask["label"])
        query_mask = np.asarray([MyCommon.rle_to_mask(one)
                                 for one in image_label_mask["masks"]], dtype=np.int8)
        return query_img, query_label, query_mask

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_label, query_mask, support_imgs, support_labels_c, \
        support_masks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_label = torch.tensor(np.array(query_label))
        query_mask = torch.tensor(np.array(query_mask))

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        support_labels_c = [torch.tensor(np.array(one)) for one in support_labels_c]
        support_masks = [torch.tensor(np.array(one)) for one in support_masks]
        support_masks = torch.stack(support_masks)

        query_label = F.interpolate(query_label.unsqueeze(0).unsqueeze(0).float(),
                                    query_img.size()[-2:], mode='nearest').squeeze()
        query_label, query_ignore_idx = self.extract_ignore_idx(query_label.float(), class_sample)

        support_labels, support_ignore_idxs = [], []
        for slabel in support_labels_c:
            slabel = F.interpolate(slabel.unsqueeze(0).unsqueeze(0).float(),
                                   support_imgs.size()[-2:], mode='nearest').squeeze()
            support_label, support_ignore_idx = self.extract_ignore_idx(slabel, class_sample)
            support_labels.append(support_label)
            support_ignore_idxs.append(support_ignore_idx)
        support_labels = torch.stack(support_labels)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_label': query_label,
                 'query_mask': query_mask[:self.mask_num],
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_labels': support_labels,
                 'support_masks': support_masks[:,:self.mask_num],
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'class_id': torch.tensor(class_sample)}
        return batch

    def extract_ignore_idx(self, label, class_id):
        # label_np = np.array(label)
        # print(f"[DEBUG] class_id: {class_id}, expected pixel: {class_id + 1}")
        # print(f"[DEBUG] label unique pixel values: {np.unique(label_np)}")
        boundary = (label / 255).floor()
        label[label != class_id + 1] = 0
        label[label == class_id + 1] = 1

        return label, boundary

    def load_frame(self, query_name, support_names):
        if self.use_mask:
            query_img, query_label, query_mask = self.get_data_and_mask(query_name)
            _support_data = [self.get_data_and_mask(name) for name in support_names]
            support_imgs = [one[0] for one in _support_data]
            support_labels = [one[1] for one in _support_data]
            support_masks = [one[2] for one in _support_data]
        else:
            query_img = self.read_img(query_name)
            query_label = self.read_label(query_name)
            query_mask = self.read_label(query_name)
            support_imgs = [self.read_img(name) for name in support_names]
            support_labels = [self.read_label(name) for name in support_names]
            support_masks = [self.read_label(name) for name in support_names]
            pass
        return query_img, query_label, query_mask, \
               support_imgs, support_labels, support_masks, query_img.size

    def read_label(self, img_name):
        path = os.path.join(self.ann_path, img_name[:-2], img_name) + '.png'
        label_img = Image.open(path)
        label_np = np.array(label_img)  # 保留调色板索引作为类别
        return Image.fromarray(label_np.astype(np.uint8))  # 转成 Image，保持一致性



    def read_img(self, img_name):
        return Image.open(os.path.join(self.img_path, img_name[:-2], img_name) + '.tif')

    def sample_episode(self, idx):
        if self.split == "train":
            class_sample = self.class_ids[idx % len(self.class_ids)]
            query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        else:
            query_name, class_sample = self.img_metadata[idx]
            pass

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break
            pass
        return query_name, support_names, class_sample

    def build_class_ids(self):
        nclass_train = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_train + i for i in range(nclass_train)]
        class_ids_train = [x for x in range(self.nclass) if x not in class_ids_val]
        return class_ids_train if self.split == 'train' else class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            split = "trn" if split == "train" else split
            fold_n_metadata = os.path.join('data/splits/LCML/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
                ####
            for data in fold_n_metadata:
                if '__' not in data:
                    print("Invalid filename:", data)
                ####
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'train':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)
        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise

    pass

# class DatasetLCML(Dataset):

#     def __init__(self, datapath, fold, transform, split, shot, use_mask=False, mask_num=128):
#         self.split = 'val' if split in ['val', 'test'] else 'train'
#         self.fold = fold
#         self.nclass = 16
#         self.benchmark = 'lcml'
#         self.shot = shot
#         self.datapath = datapath
#         self.use_mask = use_mask # 是否使用预提取的 mask（如 SAM
#         self.mask_num = mask_num # 使用的最大 mask 数量

#         self.img_path = os.path.join(datapath, 'Images')
#         self.ann_path = os.path.join(datapath, 'Labels')
#         self.pkl_path = os.path.join(datapath, 'sam_mask_vit_h_t64_32_s50')

#         self.transform = transform
        
#         self.img_metadata = self.build_img_metadata()
#         self.class_ids = self.build_class_ids()
#         self.img_metadata_classwise = self.build_img_metadata_classwise()
#         pass

#     def get_data_and_mask(self, pkl_name):
#         pkl = os.path.join(self.pkl_path, pkl_name + '.pkl')
#         with open(pkl, "rb") as f:
#             image_label_mask = pickle.load(f)

#         query_img = Image.fromarray(image_label_mask["image"])
#         query_label = Image.fromarray(image_label_mask["label"])
#         query_mask = np.asarray([MyCommon.rle_to_mask(one)
#                                  for one in image_label_mask["masks"]], dtype=np.int8)
#         return query_img, query_label, query_mask

#     def __len__(self):
#         return len(self.img_metadata)

#     def __getitem__(self, idx):
#         idx %= len(self.img_metadata)  # for testing, as n_images < 1000
#         query_name, support_names, class_sample = self.sample_episode(idx)
#         query_img, query_label, query_mask, support_imgs, support_labels_c, \
#         support_masks, org_qry_imsize = self.load_frame(query_name, support_names)

#         query_img = self.transform(query_img)
#         query_label = torch.tensor(np.array(query_label))
#         query_mask = torch.tensor(np.array(query_mask))

#         support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
#         support_labels_c = [torch.tensor(np.array(one)) for one in support_labels_c]
#         support_masks = [torch.tensor(np.array(one)) for one in support_masks]
#         support_masks = torch.stack(support_masks)

#         query_label = F.interpolate(query_label.unsqueeze(0).unsqueeze(0).float(),
#                                     query_img.size()[-2:], mode='nearest').squeeze()
#         query_label, query_ignore_idx = self.extract_ignore_idx(query_label.float(), class_sample)

#         support_labels, support_ignore_idxs = [], []
#         for slabel in support_labels_c:
#             slabel = F.interpolate(slabel.unsqueeze(0).unsqueeze(0).float(),
#                                    support_imgs.size()[-2:], mode='nearest').squeeze()
#             support_label, support_ignore_idx = self.extract_ignore_idx(slabel, class_sample)
#             support_labels.append(support_label)
#             support_ignore_idxs.append(support_ignore_idx)
#         support_labels = torch.stack(support_labels)
#         support_ignore_idxs = torch.stack(support_ignore_idxs)

#         batch = {'query_img': query_img,
#                  'query_label': query_label,
#                  'query_mask': query_mask[:self.mask_num],
#                  'query_name': query_name,
#                  'query_ignore_idx': query_ignore_idx,

#                  'org_query_imsize': org_qry_imsize,

#                  'support_imgs': support_imgs,
#                  'support_labels': support_labels,
#                  'support_masks': support_masks[:,:self.mask_num],
#                  'support_names': support_names,
#                  'support_ignore_idxs': support_ignore_idxs,

#                  'class_id': torch.tensor(class_sample)}


#         return batch

#     def extract_ignore_idx(self, label, class_id):
#         # label_np = np.array(label)
#         # print(f"[DEBUG] class_id: {class_id}, expected pixel: {class_id + 1}")
#         # print(f"[DEBUG] label unique pixel values: {np.unique(label_np)}")
        
#         label[label != class_id + 1] = 0
#         label[label == class_id + 1] = 1
#         boundary = (label / 255).floor()
#         return label, boundary

#     def load_frame(self, query_name, support_names):
#         if self.use_mask:
#             query_img, query_label, query_mask = self.get_data_and_mask(query_name)
#             _support_data = [self.get_data_and_mask(name) for name in support_names]
#             support_imgs = [one[0] for one in _support_data]
#             support_labels = [one[1] for one in _support_data]
#             support_masks = [one[2] for one in _support_data]
#         else:
#             query_img = self.read_img(query_name)
#             query_label = self.read_label(query_name)
#             query_mask = self.read_label(query_name)
#             support_imgs = [self.read_img(name) for name in support_names]
#             support_labels = [self.read_label(name) for name in support_names]
#             support_masks = [self.read_label(name) for name in support_names]
#             pass
#         return query_img, query_label, query_mask, \
#                support_imgs, support_labels, support_masks, query_img.size

#     def read_label(self, img_name):
#         """
#         安全读取标签 PNG，保持调色板索引原样。
#         返回 PIL Image，以便后续转换和处理。
#         """
#         # 构建路径
#         path = os.path.join(self.ann_path, img_name[:-2], img_name) + '.png'
        
#         # 读取图像
#         label_img = Image.open(path)
        
#         # 如果是 P 模式，直接读取调色板索引
#         if label_img.mode == 'P':
#             label_np = np.array(label_img, dtype=np.uint8)
#         else:
#             # 防护：非 P 模式也转为整数索引
#             label_np = np.array(label_img.convert('L'), dtype=np.uint8)
        
#         # 返回 PIL Image，保持一致性
#         return Image.fromarray(label_np.astype(np.uint8))



#     def read_img(self, img_name):
#         return Image.open(os.path.join(self.img_path, img_name[:-2], img_name) + '.tif')

#     def sample_episode(self, idx):
#         if self.split == "train":
#             class_sample = self.class_ids[idx % len(self.class_ids)]
#             query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
#         else:
#             query_name, class_sample = self.img_metadata[idx]
#             pass

#         support_names = []
#         while True:  # keep sampling support set if query == support
#             support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
#             if query_name != support_name: support_names.append(support_name)
#             if len(support_names) == self.shot: break
#             pass
#         return query_name, support_names, class_sample

#     def build_class_ids(self):
#         """
#         根据 txt 文件中实际出现的类别，构建 class_ids 列表
#         """
#         class_ids = sorted(list(set([cls for _, cls in self.img_metadata])))
#         return class_ids


#     def build_img_metadata(self):
#         split_file = os.path.join(self.datapath, f"{self.split}.txt")
#         img_metadata = []
#         with open(split_file, 'r') as f:
#             lines = f.read().splitlines()
#             for line in lines:
#                 name, cls = line.split('__')
#                 cls = int(cls)

#                 if self.split == "train":
#                     # 只保留 1-15 类
#                     if 1 <= cls <= 15:
#                         img_metadata.append([name, cls - 1])  # 类别从 0 开始编号
#                 else:  # val/test
#                     # 只保留 16 类
#                     if cls == 16:
#                         img_metadata.append([name, cls - 1])  # 仍然减 1 → 类 id = 15
#         print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
#         return img_metadata



#     def build_img_metadata_classwise(self):
#         """
#         构建按类别索引的图像字典，保证每个类别都有列表。
#         对训练集，包含类 0~14
#         对验证集，包含类 15（即原16类）
#         """
#         # 获取当前数据集的所有类别
#         all_classes = sorted(list(set([cls for _, cls in self.img_metadata])))
        
#         # 初始化字典
#         img_metadata_classwise = {cls: [] for cls in all_classes}
        
#         # 遍历当前数据集的 img_metadata
#         for img_name, img_class in self.img_metadata:
#             img_metadata_classwise[img_class].append(img_name)
        
#         # 对于验证集可能不存在训练集样本的类别，也初始化为空列表
#         if self.split != "train":
#             for cls in range(self.nclass):
#                 if cls not in img_metadata_classwise:
#                     img_metadata_classwise[cls] = []
        
#         return img_metadata_classwise
        

    
class FSSDataset(object):

    @classmethod
    def initialize(cls, img_size, datapath):
        cls.datasets = {'isaid': DatasetISAID, 'dlrsd': DatasetDLRSD, 'lcml': DatasetLCML}

        # 普通 3 通道数据集（RGB）均值/标准差
        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std  = [0.229, 0.224, 0.225]

        # LCML 5 波段数据集均值/标准差
        cls.img_mean_lcml = [0.209, 0.394, 0.380, 0.344, 0.398]
        cls.img_std_lcml  = [0.141, 0.027, 0.032, 0.046, 0.108]

        # LCML 专用 transform


        # 默认 transform（RGB 数据集）
        cls.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(cls.img_mean, cls.img_std)
        ])

        cls.datapath = datapath

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1, use_mask=False, mask_num=128):
        shuffle = split == 'train'
        transform = cls.transform
        # 根据数据集选择 transform
        # if benchmark == 'lcml':
        #     transform = cls.transform_lcml
        # else:
        #     transform = cls.transform

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=transform, split=split,
                                          shot=shot, use_mask=use_mask, mask_num=mask_num)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)
        return dataloader