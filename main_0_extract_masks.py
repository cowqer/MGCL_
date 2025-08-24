import os
import cv2
import sys
import torch
import pickle
import platform
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
from alisuretool.Tools import Tools
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
sys.path.append("../segment-anything-main")

from segment_anything import sam_model_registry
from extractmasks.my_utils import my_masked_average_pooling
from extractmasks.my_data.utils_rle import mask_to_rle_pytorch, rle_to_mask
from extractmasks.my_automatic_mask_generator2 import SamAutomaticMaskGenerator
from extractmasks.my_transforms import MyResizeLongestSide


class Config(object):
    dataset_name = "said"
    use_gpu = True
    gpu_id = "0"
    model_type = "vit_h"
    target_size = 64
    points_per_side = 32
    stability_score_thresh = 0.50
    data_root = "/data/seekyou/Data/7class_LCML/Maping4DLRSD"
    data_sam_mask_name = "sam_mask_{}_t{}_p{}_s{}".format(
            model_type, target_size, points_per_side, int(stability_score_thresh * 100))
    # need to download from https://github.com/facebookresearch/segment-anything#model-checkpoints
    pretrain_checkpoint = "./segment-anything-main/checkpoints/sam_vit_h_4b8939.pth"
    pass

    @staticmethod
    def gpu_setup(use_gpu, gpu_id):
        if torch.cuda.is_available() and use_gpu:
            Tools.print()
            Tools.print('Cuda available with GPU: {}'.format(torch.cuda.get_device_name(0)))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            device = torch.device("cuda:{}".format(gpu_id))
        else:
            Tools.print()
            Tools.print('Cuda not available')
            device = torch.device("cpu")
        return device

    pass

class DLRSDAll(Dataset):

    def __init__(self, split=None):
        # 遍历所有类别子目录，获取图片和对应标签路径
        image_dirs = sorted(glob(os.path.join(Config.data_root, "Images")))
        label_dirs = sorted(glob(os.path.join(Config.data_root, "Labels")))

        self.all_image = []
        self.all_label = []

        for image_dir, label_dir in zip(image_dirs, label_dirs):
            image_files = sorted(glob(os.path.join(image_dir, "*.tif")))  # 根据实际格式换成 *.jpg 或 *.tif
            label_files = sorted(glob(os.path.join(label_dir, "*.png")))
            self.all_image.extend(image_files)
            self.all_label.extend(label_files)

        self._check()

    def _check(self):
        assert len(self.all_image) == len(self.all_label), "Mismatch between images and labels"
        for img_path, lbl_path in zip(self.all_image, self.all_label):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_name = os.path.splitext(os.path.basename(lbl_path))[0]
            assert img_name == lbl_name, f"Name mismatch: {img_name} vs {lbl_name}"

    def __len__(self):
        return len(self.all_image)

    def read_img(self, image_path):
        return Image.open(image_path)

    def __getitem__(self, idx):
        image_path = self.all_image[idx]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)##用OpenCV读RGB图像
        label_path = self.all_label[idx] ##读标签PNG图像为数组
        # label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2GRAY)
        label = np.array(Image.open(label_path))
        #返回字典
        return {'image': image, "label": label,
                'image_path': image_path, 'label_path': label_path}

    pass

class RunnerSAMForExtractingMask(object):#主执行类，负责生成并保存掩膜

    def __init__(self, is_dao=False, split="val"):
        self.is_dao = is_dao
        self.split = split
        self.target_size = Config.target_size
        self.device = Config.gpu_setup(use_gpu=Config.use_gpu, gpu_id=Config.gpu_id)
        self.sam = sam_model_registry[Config.model_type](checkpoint=Config.pretrain_checkpoint).to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam, points_per_side=Config.points_per_side,
            stability_score_thresh=Config.stability_score_thresh)
        self.dataset = DLRSDAll(split=self.split)
        pass

    @torch.no_grad()
    def runner_1_extracted_mask(self, result_mask_name, mask_num=128, target_size=64):
        for i in tqdm(range(len(self.dataset))):##tqdm进度条显示
            # 读取数据
            which = (len(self.dataset) - i - 1) if self.is_dao else i ##如果是DAO模式，则从后往前读取数据
            data_one = self.dataset.__getitem__(which)
            image, image_path = data_one["image"], data_one["image_path"] # 取出图片和标签
            label, label_path = data_one["label"], data_one["label_path"]

            # 是否已经存在特征
            now_name = os.path.splitext(os.path.basename(image_path))[0]
            result_mask_file = Tools.new_dir(os.path.join(
                Config.data_root, result_mask_name, "{}.pkl".format(now_name))) # 生成保存路径
            if os.path.exists(result_mask_file):
                Tools.print("File exist: {}".format(result_mask_file))
                continue

            # 提取特征
            masks = self.mask_generator.generate(image)
            masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
            for one in masks:
                one["segmentation"] = mask_to_rle_pytorch(torch.tensor(one["segmentation"][None, :]))[0] #对mask做RLE编码
                pass
            masks, masks_target = self._deal_feature( #处理并resize所有mask
                masks=masks, mask_num=mask_num, target_size=target_size)
            #打包保存内容
            now_data = {"input_size": (1024, 1024), "image_size": image.shape[:2],
                        "image": image, "image_name": now_name, "label": label,
                        "label_target": MyResizeLongestSide(
                            target_length=64).apply_mask(label, is_pad=True),
                        "masks_target": masks_target, "masks": masks}

            # 保存结果
            with open(result_mask_file, "wb") as f:
                pickle.dump(now_data, f)
                pass
            pass
        pass

    @classmethod
    def _deal_feature(cls, masks, mask_num=128, target_size=64):
        ########################################################################
        for mask in masks:
            mask["segmentation"] = rle_to_mask(mask["segmentation"])# 解码RLE
            pass

        masks = cls._check_mask(masks, mask_num=mask_num) # 检查mask数量，确保有mask_num个
        masks_target = cls._transform_to_target_size(masks=masks, target_size=target_size)# 将mask转换为目标大小
        ########################################################################

        ########################################################################
        #拼接为tensor
        masks = np.concatenate([mask_one["segmentation"][np.newaxis, :, :]
                                for mask_one in masks], axis=0)
        masks_target = np.concatenate([mask_one["segmentation"][np.newaxis, :, :]
                                       for mask_one in masks_target], axis=0)
        ########################################################################
        #重新编码RLE
        masks = mask_to_rle_pytorch(torch.tensor(masks))
        masks_target = mask_to_rle_pytorch(torch.tensor(masks_target))
        return masks, masks_target

    @staticmethod
    def _transform_to_target_size(masks, target_size):#对单个mask做resize和bbox点位变换
        transform_mask = MyResizeLongestSide(target_length=target_size)

        mask_target = []
        for mask in masks:
            current_mask = {}
            size_x, size_y = mask["crop_box"][-2], mask["crop_box"][-1]
            mask["segmentation"] = np.asarray(mask["segmentation"], dtype=np.uint8)
            current_mask["segmentation"] = transform_mask.apply_mask(mask["segmentation"], is_pad=True)
            current_mask["area"] = np.sum(current_mask["segmentation"])
            current_mask["bbox"] = transform_mask.apply_xs(mask["bbox"], size_x=size_x, size_y=size_y)
            current_mask["crop_box"] = transform_mask.apply_xs(mask["crop_box"], size_x=size_x, size_y=size_y)
            current_mask["point_coords"] = [transform_mask.apply_xs(one, size_x=size_x, size_y=size_y)
                                            for one in mask["point_coords"]]
            current_mask["predicted_iou"] = mask["predicted_iou"]
            current_mask["stability_score"] = mask["stability_score"]
            mask_target.append(current_mask)
            pass

        return mask_target

    @staticmethod
    def _check_mask(masks, mask_num):
        if len(masks) > mask_num:
            masks = masks[:mask_num]
        else:
            if len(masks) > 0:
                while len(masks) < mask_num:
                    current_mask = {
                        "segmentation": np.zeros_like(masks[0]["segmentation"]),
                        "area": 0,
                        "bbox": [0, 0, 0, 0],
                        "predicted_iou": 0.0,
                        "point_coords": [[0.0, 0.0]],
                        "stability_score": 0,
                        "crop_box": masks[0]["crop_box"],
                        "ratio": 0.0,
                    }
                    masks.append(current_mask)
                pass
            pass
        return masks

    @staticmethod
    def runner_2_vis_mask(mask_path, result_path):# 从pkl文件里读出来保存可视化图片
        if os.path.exists(mask_path):
            try:
                with open(mask_path, "rb") as f:
                    result = pickle.load(f)
                    image = result["image"]
                    label = result["label"]
                    masks = result["masks"]

                    result_path = Tools.new_dir(result_path)
                    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
                        os.path.join(result_path, "0_image.jpg"))
                    Image.fromarray(np.asarray(label * (255 // 15), dtype=np.uint8)).save(
                        os.path.join(result_path, "0_label.png"))

                    for index, ann in enumerate(masks):
                        resul_file = os.path.join(result_path, f"{index+1}.bmp")
                        m = rle_to_mask(ann)
                        Image.fromarray(np.asarray(np.asarray(m, dtype=np.int32) * 255,
                                                   dtype=np.uint8)).save(resul_file)
                        pass
                    pass
            except Exception:
                Tools.print(f"Error file {mask_path}")
        else:
            Tools.print(f"No file {mask_path}")
        pass

    pass


if __name__ == '__main__':
    runner = RunnerSAMForExtractingMask(is_dao=False, split="train")
    runner.runner_1_extracted_mask(result_mask_name=Config.data_sam_mask_name)
    pass