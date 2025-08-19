import torch
import argparse
from alisuretool.Tools import Tools
from net import *
from util.util_tools import MyCommon, AverageMeter, Logger
from dataset.dataset_tools import FSSDataset, Evaluator
import yaml
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

palette = {
    1: [166, 202, 240], 2: [128,128,0], 3: [0,0,128], 4: [255,0,0],
    5: [0,128,0], 6: [128,0,0], 7: [255,233,233], 8: [160,160,164],
    9: [0,128,128], 10: [90,87,255], 11: [255,255,0], 12: [255,192,0],
    13: [0,0,255], 14: [255,0,192], 15: [128,0,128], 16: [0,255,0],
    17: [0,255,255], 18: [255,215,0]
}

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    args = argparse.Namespace(**cfg)
    yaml_base = os.path.splitext(os.path.basename(yaml_path))[0]
    args.logpath = yaml_base
    return args

def save_pred_masks(pred, save_dir="./pred_masks"):
    """
    pred: torch.Tensor, shape (B,H,W) or (B,num_classes,H,W)
    """
    os.makedirs(save_dir, exist_ok=True)

    if pred.dim() == 4:  # (B, num_classes, H, W)
        pred = torch.argmax(pred, dim=1)  # -> (B,H,W)

    pred = pred.cpu().numpy().astype(np.uint8)

    for i in range(pred.shape[0]):
        mask = pred[i]  # (H,W)
        img = Image.fromarray(mask)  # 保存成灰度图，每个像素=类别id
        img.save(os.path.join(save_dir, f"pred_{i}.png"))
        
class Runner(object):

    def __init__(self, args):
        self.args = args
        self.device = MyCommon.gpu_setup(use_gpu=self.args.use_gpu, gpu_id=args.gpuid)

        net_cls = globals()[self.args.net_name]
        self.model = net_cls(args).to(self.device)
        self.model.eval()
        weights = torch.load(args.load, map_location=None if self.args.use_gpu else torch.device('cpu'))
        weights = {one.replace("module.", ""): weights[one] for one in weights.keys()}
        weights = {one.replace("hpn_learner.", "mgcd."): weights[one] for one in weights.keys()}
        self.model.load_state_dict(weights)

        FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath)
        self.dataloader_val = FSSDataset.build_dataloader(
            args.benchmark, args.bsz, args.nworker, args.fold, 'val', args.shot,
            use_mask=args.mask, mask_num=args.mask_num)
        
    @torch.no_grad()
    def test(self):
        dataloader = self.dataloader_val
        average_meter = AverageMeter(dataloader.dataset, device=self.device)

        for idx, batch in enumerate(dataloader):
            batch = MyCommon.to_cuda(batch, device=self.device)
            pred = self.model.predict_nshot(batch)

            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
            average_meter.write_process(idx, len(dataloader), 0, write_batch_idx=5)
        miou, fb_iou = average_meter.compute_iou()
        return miou, fb_iou
    

    @torch.no_grad()
    def test_class(self):
        dataloader = self.dataloader_val
        average_meter = AverageMeter(dataloader.dataset, device=self.device)

        for idx, batch in enumerate(dataloader):
            batch = MyCommon.to_cuda(batch, device=self.device)
            # for k in batch.keys():
            #     print(k)
            pred = self.model.predict_nshot(batch)
            
            Evaluator.visualize_batch_mask_only(pred, batch, save_dir="./vis_results", target_classes=[16,17])

            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
            average_meter.write_process(idx, len(dataloader), 0, write_batch_idx=5)
        miou, fb_iou, iou, class_ids = average_meter.compute_iou_class()
        return miou, fb_iou, iou, class_ids

if __name__ == '__main__':
    
    import sys
    
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python test.py <config.yaml> <model_weights.pt> [shot]")
        exit(1)

    yaml_path = sys.argv[1]
    model_path = sys.argv[2]
    shot_val = int(sys.argv[3]) if len(sys.argv) == 4 else 1  # 默认1

    args = load_yaml_config(yaml_path)
    args.load = model_path  # 覆盖权重路径
    args.shot = shot_val    # 用第三个参数覆盖shot

    # 设置默认参数缺失（防止yaml中没写net_name）
    if not hasattr(args, 'net_name'):
        Tools.print("Warning: net_name not specified in config! Plz check with your mzf eyes, using default.")
        args.net_name = 'YourDefaultNetClassName'  # 请替换为你的默认网络类名

    MyCommon.fix_randseed(0)
    Logger.initialize(args, training=False)
    runner = Runner(args=args)
    Logger.log_params(runner.model)

    miou, fb_iou, iou, class_ids = runner.test_class()
    
    # runner.test_vis(save_dir="./vis_results")
    
    print("mIoU (mean):", miou.item() if torch.is_tensor(miou) else miou)

    # Tools.print("FB-IoU:", fb_iou)
    print("FB-IoU:", fb_iou.item() if torch.is_tensor(fb_iou) else fb_iou)
    Tools.print("Per-class IoU:")
    
# dataloader_val 是验证集的 dataloader
    id2name = runner.dataloader_val.dataset.id2name
    print("ID to Name Mapping:", id2name)
  

    Tools.print("Per-class IoU:")
    for class_id, class_iou in zip(class_ids.tolist(), iou):
        class_name = id2name.get(class_id, f"Class_{class_id}")
        print(f"{class_name} (id {class_id}): {class_iou.item():.4f}")




# 这样就能看到每一类的 mIoU。


    try:
        subprocess.run(["python", "./logs/organize_logs.py"], check=True)
        Tools.print("✅ 日志整理完成！")
    except subprocess.CalledProcessError as e:
        Tools.print("❌ 日志整理失败：", e)
