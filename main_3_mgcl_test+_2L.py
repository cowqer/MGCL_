import torch
import argparse
from alisuretool.Tools import Tools
from net import *
from util.util_tools import MyCommon, AverageMeter, Logger
from dataset.dataset_tools import FSSDataset, Evaluator
import yaml
import os
import subprocess
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
# from torchsummary import summary


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
        Logger.log_params(self.model)
        
    @torch.no_grad()
    def test(self):
        dataloader = self.dataloader_val
        average_meter = AverageMeter(dataloader.dataset, device=self.device)

        for idx, batch in enumerate(dataloader):
            batch = MyCommon.to_cuda(batch, device=self.device)
            coarse_logit, refined_logit = self.model.predict_nshot(batch)

            area_inter, area_union = Evaluator.classify_prediction(refined_logit, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
            average_meter.write_process(idx, len(dataloader), 0, write_batch_idx=5)
        miou, fb_iou = average_meter.compute_iou()
        return miou, fb_iou
    
    @torch.no_grad()
    def test_class(self, target_classes=None):
        dataloader = self.dataloader_val
        average_meter = AverageMeter(dataloader.dataset, device=self.device)

        for idx, batch in enumerate(dataloader):
            batch = MyCommon.to_cuda(batch, device=self.device)
            logit = self.model.predict_nshot(batch)

            # 只有指定了 --class 时才可视化
            if target_classes is not None:
                Evaluator.visualize_batch_mask_only(
                    logit, batch, save_dir="./vis_results", target_classes=target_classes
                )

            area_inter, area_union = Evaluator.classify_prediction(logit, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
            average_meter.write_process(idx, len(dataloader), 0, write_batch_idx=10)

        miou, fb_iou, iou, class_ids = average_meter.compute_iou_class()
        return miou, fb_iou, iou, class_ids


if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="YAML 配置文件路径")
    parser.add_argument("weights", type=str, help="模型权重路径")
    parser.add_argument("shot", type=int, nargs="?", default=None, help="few-shot 数量（可选）")
    parser.add_argument("--class", type=int, nargs="+", dest="vis_classes", default=None,
                        help="指定需要可视化的类别ID (例如: --class 7 8)")
    args_cli = parser.parse_args()

    # 加载 yaml
    args = load_yaml_config(args_cli.config)
    args.load = args_cli.weights  # 覆盖权重路径

    # shot 参数优先级: CLI > yaml
    if args_cli.shot is not None:
        args.shot = args_cli.shot

    # net_name 默认设置
    if not hasattr(args, 'net_name'):
        Tools.print("Warning: net_name not specified in config! Using default.")
        args.net_name = 'YourDefaultNetClassName'

    MyCommon.fix_randseed(0)
    Logger.initialize(args, training=False)
    runner = Runner(args=args)
    Logger.log_params(runner.model)
    # Tools.print("Model Architcture:\n")
    # Tools.print(str(runner.model))

    # 调用 test_class（始终算 IoU），是否可视化由 --class 控制
    miou, fb_iou, iou, class_ids = runner.test_class(target_classes=args_cli.vis_classes)

    # 输出整体指标
    # 输出整体指标
    print("mIoU (mean):", miou.item() if torch.is_tensor(miou) else miou)
    print("FB-IoU:", fb_iou.item() if torch.is_tensor(fb_iou) else fb_iou)

    # 输出每类指标
    Tools.print("Per-class IoU:")
    id2name = runner.dataloader_val.dataset.id2name
    results = {
        "mIoU": miou.item() if torch.is_tensor(miou) else miou,
        "FB-IoU": fb_iou.item() if torch.is_tensor(fb_iou) else fb_iou,
        "per_class": {}
    }
    for class_id, class_iou in zip(class_ids.tolist(), iou):
        class_name = id2name.get(class_id, f"Class_{class_id}")
        results["per_class"][f"{class_name}_id{class_id}"] = class_iou.item()

        print(f"{class_name} (id {class_id}): {class_iou.item():.4f}")

    # === 保存结果 ===
    import json, os
    save_dir = os.path.dirname(args.load)   # pt 文件所在目录
    save_name = os.path.splitext(os.path.basename(args.load))[0] + "_eval.json"
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    Tools.print(f"📂 结果已保存到 {save_path}")


    # 日志整理
    try:
        subprocess.run(["python", "./logs/organize_logs.py"], check=True)
        Tools.print("✅ 日志整理完成！")
    except subprocess.CalledProcessError as e:
        Tools.print("❌ 日志整理失败：", e)
