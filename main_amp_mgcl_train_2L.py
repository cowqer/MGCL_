import torch 
import argparse
from torch import amp
from alisuretool.Tools import Tools
from net import *
from dataset.dataset_tools import FSSDataset, Evaluator
from util.util_tools import MyCommon, MyOptim, AverageMeter, Logger
import yaml
import datetime
import os
import shutil

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    args = argparse.Namespace(**cfg)
    yaml_base = os.path.splitext(os.path.basename(yaml_path))[0]
    now = datetime.datetime.now()
    time_str = now.strftime("%m%d__%H_%M%S")

    # 添加 amp 标识
    args.logpath = os.path.join(yaml_base, f"{yaml_base}_amp_{time_str}")
    args.loggerpath = os.path.join('logs', args.logpath + '.log')

    print(f"Log path: {args.logpath}")
    Logger.initialize(args, training=True)
    return args



class Runner(object):

    def __init__(self, args):
        self.args = args
        self.device = MyCommon.gpu_setup(use_gpu=True, gpu_id=args.gpuid)

        net_cls = globals()[self.args.net_name] if hasattr(self.args, "net_name") else myNetwork
        self.model = net_cls(args).to(self.device)
        self.optimizer = MyOptim.get_finetune_optimizer(args, self.model)
        Logger.info("Model architecture:\n{}".format(self.model))
        FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath)
        self.dataloader_train = FSSDataset.build_dataloader(
            args.benchmark, args.bsz, 8, args.fold, 'train', use_mask=args.mask, mask_num=args.mask_num)
        self.dataloader_val = FSSDataset.build_dataloader(
            args.benchmark, args.bsz, 8, args.fold, 'val', use_mask=args.mask, mask_num=args.mask_num)
        os.makedirs(args.loggerpath, exist_ok=True)
        shutil.copy(os.path.join("net", "FBC.py"), os.path.join(args.loggerpath, "FBC.py"))
        shutil.copy(os.path.join("net", "CD.py"), os.path.join(args.loggerpath, "CD.py"))
        shutil.copy(os.path.join("net", "baseline_.py"), os.path.join(args.loggerpath, "baseline_.py"))
        Logger.log_params(self.model)

        # 初始化 GradScaler
        self.scaler = amp.GradScaler("cuda")

    def train(self):
        best_val_miou = 0
        for epoch in range(self.args.epoch_num):

            # Tools.print(f"[{now_str}] begin epoch {epoch} train")
            Logger.info(f"===== [Epoch {epoch}] Begin Train =====")
            MyOptim.adjust_learning_rate_poly(self.args, self.optimizer, epoch, self.args.epoch_num)

            train_loss, train_miou, train_fb_iou = self.train_one_epoch(epoch, self.dataloader_train)

            # Tools.print(f"[{now_str}] begin epoch {epoch} test")
            
            Logger.info(f"===== [Epoch {epoch}] Begin Validation =====")
            with torch.no_grad():
                val_loss, val_miou, val_fb_iou = self.test_one_epoch(epoch, self.dataloader_val)

            if val_miou > best_val_miou:
                best_val_miou = val_miou
                Logger.save_model_miou(self.model, epoch, val_miou)

            Logger.tbd_writer.add_scalars('data/loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'train_miou': train_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'train_fb_iou': train_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')

    def train_one_epoch(self, epoch, dataloader):
        self.model.train()
        average_meter = AverageMeter(dataloader.dataset, device=self.device)
        for idx, batch in enumerate(dataloader):
            batch = MyCommon.to_cuda(batch, device=self.device)
            self.optimizer.zero_grad()

            with amp.autocast("cuda"):  # AMP 前向
                coarse_logit, refined_logit = self.model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_labels'].squeeze(1),
                                   query_mask=batch['query_mask'] if 'query_mask' in batch else None,
                                   support_masks=batch['support_masks'].squeeze(1) if 'support_masks' in batch else None)
                loss1 = self.model.compute_objective(coarse_logit, batch['query_label'])
                loss2 = self.model.compute_objective(refined_logit, batch['query_label'])
                loss = 0.5 * loss1 + loss2
            # AMP 反向
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            pred = refined_logit.argmax(dim=1)
            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

        average_meter.write_result('Training', epoch)
        avg_loss = MyCommon.mean(average_meter.loss_buf)
        miou, fb_iou = average_meter.compute_iou()
        return avg_loss, miou, fb_iou

    def test_one_epoch(self, epoch, dataloader):
        self.model.eval()
        Tools.print(f"[Epoch {epoch}] Evaluating with network: {self.model.__class__.__name__}")
        average_meter = AverageMeter(dataloader.dataset, device=self.device)
        for idx, batch in enumerate(dataloader):
            batch = MyCommon.to_cuda(batch, device=self.device)
            with amp.autocast("cuda"):
                coarse_logit, refined_logit  = self.model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_labels'].squeeze(1),
                                   query_mask=batch['query_mask'] if 'query_mask' in batch else None,
                                   support_masks=batch['support_masks'].squeeze(1) if 'support_masks' in batch else None)
                loss1 = self.model.compute_objective(refined_logit, batch['query_label'])
                loss2 = self.model.compute_objective(coarse_logit, batch['query_label'])
                loss = 0.5*loss1 + loss2
                
            pred = refined_logit.argmax(dim=1)
            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

        average_meter.write_result('Validation', epoch)
        avg_loss = MyCommon.mean(average_meter.loss_buf)
        miou, fb_iou = average_meter.compute_iou()
        return avg_loss, miou, fb_iou


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <config.yaml>")
        exit(1)

    yaml_path = sys.argv[1]
    args = load_yaml_config(yaml_path)
    runner = Runner(args=args)
    runner.train()
