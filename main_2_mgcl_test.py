import torch
import argparse
from alisuretool.Tools import Tools
from net import *
from util.util_tools import MyCommon, AverageMeter, Logger
from dataset.dataset_tools import FSSDataset, Evaluator
import yaml
import os

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    import argparse
    args = argparse.Namespace(**cfg)
    # 获取yaml文件名（不含扩展名）
    yaml_base = os.path.splitext(os.path.basename(yaml_path))[0]
    args.logpath = yaml_base
    return args


class Runner(object):

    def __init__(self, args):
        self.args = args
        self.device = MyCommon.gpu_setup(use_gpu=self.args.use_gpu, gpu_id=args.gpuid)

        # 根据net_name动态选择网络
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
        pass

    @torch.no_grad()
    def test(self):
        dataloader = self.dataloader_val
        average_meter = AverageMeter(dataloader.dataset, device=self.device)

        for idx, batch in enumerate(dataloader):
            # 1. forward pass
            batch = MyCommon.to_cuda(batch, device=self.device)
            pred = self.model.predict_nshot(batch)

            # 2. Evaluate prediction
            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
            average_meter.write_process(idx, len(dataloader), 0, write_batch_idx=5)
            pass
        miou, fb_iou = average_meter.compute_iou()
        return miou, fb_iou

    @torch.no_grad()
    def test_class(self):
        dataloader = self.dataloader_val
        average_meter = AverageMeter(dataloader.dataset, device=self.device)

        for idx, batch in enumerate(dataloader):
            # 1. forward pass
            batch = MyCommon.to_cuda(batch, device=self.device)
            pred = self.model.predict_nshot(batch)

            # 2. Evaluate prediction
            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
            average_meter.write_process(idx, len(dataloader), 0, write_batch_idx=5)
            pass
        miou, fb_iou, iou = average_meter.compute_iou_class()
        return miou, fb_iou, iou

    pass


def my_parser(fold=0, shot=1, backbone='resnet50', benchmark="isaid",
              load='./logs/demo/best_model.pt', use_gpu=False, gpu_id=0,
              bsz=2, mask=True, mask_num=128):
    datapath = None
    if benchmark == "isaid":
        datapath = '/mnt/4T/ALISURE/FSS-RS/remote_sensing/iSAID_patches'
    elif benchmark == "dlrsd":
        datapath = '/mnt/4T/ALISURE/FSS-RS/DLRSD'

    parser = argparse.ArgumentParser(description='MGCL Pytorch Implementation')

    parser.add_argument('--logpath', type=str, default='demo')
    parser.add_argument('--use_gpu', type=bool, default=use_gpu)
    parser.add_argument('--gpuid', type=int, default=gpu_id)
    parser.add_argument('--bsz', type=int, default=bsz)
    parser.add_argument('--fold', type=int, default=fold, choices=[0, 1, 2])
    parser.add_argument('--shot', type=int, default=shot, choices=[1, 5])
    parser.add_argument('--load', type=str, default=load)
    parser.add_argument('--backbone', type=str, default=backbone,
                        choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--mask', type=bool, default=mask)
    parser.add_argument('--mask_num', type=int, default=mask_num)
    parser.add_argument('--datapath', type=str, default=datapath)
    parser.add_argument('--benchmark', type=str, default=benchmark, choices=['isaid', 'dlrsd'])
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Specify YAML config file and model weights')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--load', type=str, required=True, help='Path to model weights (.pt)')
    args_cmd = parser.parse_args()
    yaml_path = args_cmd.config

    args = load_yaml_config(yaml_path)
    args.load = args_cmd.load  # 用命令行指定pt文件路径覆盖yaml中的load

    MyCommon.fix_randseed(0)
    Logger.initialize(args, training=False)
    runner = Runner(args=args)
    Logger.log_params(runner.model)

    miou, fb_iou, iou = runner.test_class()
    Tools.print("mIoU: {} FB-IoU: {} iou: {}".format(miou, fb_iou, iou))
    pass