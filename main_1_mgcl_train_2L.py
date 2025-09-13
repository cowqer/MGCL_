import torch
import argparse
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

    import argparse
    args = argparse.Namespace(**cfg)

    # 获取yaml文件名（不含扩展名）
    yaml_base = os.path.splitext(os.path.basename(yaml_path))[0]

    # 自动生成 logpath，格式如 logs/config0/config0_0725__150310/
    now = datetime.datetime.now()
    time_str = now.strftime("%m%d__%H_%M%S")
    
    # 以配置名作为目录名，再拼接时间戳
    args.logpath = os.path.join(yaml_base, f"{yaml_base}_{time_str}")

        # cls.logpath = os.path.join('logs', logpath + '.log')
    args.loggerpath = os.path.join('logs', args.logpath+ '.log')
    print(f"Log path: {args.logpath}")

    # 初始化 Logger
    Logger.initialize(args, training=True)

    return args


class Runner(object):

    def __init__(self, args):
        self.args = args
        self.device = MyCommon.gpu_setup(use_gpu=True, gpu_id=args.gpuid)

        # 动态选择网络
        net_cls = globals()[self.args.net_name] if hasattr(self.args, "net_name") else myNetwork
        self.model = net_cls(args).to(self.device)
        self.optimizer = MyOptim.get_finetune_optimizer(args, self.model)

        FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath)
        self.dataloader_train = FSSDataset.build_dataloader(
            args.benchmark, args.bsz, 8, args.fold, 'train', use_mask=args.mask, mask_num=args.mask_num)
        self.dataloader_val = FSSDataset.build_dataloader(
            args.benchmark, args.bsz, 8, args.fold, 'val', use_mask=args.mask, mask_num=args.mask_num)
        os.makedirs(args.loggerpath, exist_ok=True)
        src_fbc_path = os.path.join("net", "FBC.py")
        dst_fbc_path = os.path.join(args.loggerpath, "FBC.py")
        shutil.copy(src_fbc_path, dst_fbc_path)
        # 记录模型结构到日志
        Logger.log_params(self.model)  # 如果Logger支持


    def train(self):
        best_val_miou = 0
        for epoch in range(self.args.epoch_num):
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            Tools.print(f"[{now_str}] begin epoch {epoch} train")

            # Tools.print("begin epoch {} train".format(epoch))

            now_lr = MyOptim.adjust_learning_rate_poly(self.args, self.optimizer, epoch, self.args.epoch_num)
            Tools.print("lr={}".format(now_lr))
            train_loss, train_miou, train_fb_iou = self.train_one_epoch(epoch, self.dataloader_train)

            # Tools.print("begin epoch {} test".format(epoch))
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            Tools.print(f"[{now_str}] begin epoch {epoch} test")
            with torch.no_grad():
                val_loss, val_miou, val_fb_iou = self.test_one_epoch(epoch, self.dataloader_val)
                pass

            # Save the best model
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                Logger.save_model_miou(self.model, epoch, val_miou)
                pass

            Logger.tbd_writer.add_scalars('data/loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'train_miou': train_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'train_fb_iou': train_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()
            pass
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')
        pass

    def train_one_epoch(self, epoch, dataloader):
        has_module = hasattr(self.model, "module")
        self.model.module.train_mode() if has_module else self.model.train_mode()
        average_meter = AverageMeter(dataloader.dataset, device=self.device)

        for idx, batch in enumerate(dataloader):
            # 1. forward pass
            batch = MyCommon.to_cuda(batch, device=self.device)
            coarse_logit, refined_logit  = self.model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_labels'].squeeze(1),
                               query_mask=batch['query_mask'] if 'query_mask' in batch else None,
                               support_masks=batch['support_masks'].squeeze(1) if 'support_masks' in batch else None)
            # 2. Compute loss & update model parameters
            if has_module:
                loss1 = self.model.module.compute_objective(coarse_logit, batch['query_label'])
                loss2 = self.model.module.compute_objective(refined_logit, batch['query_label'])
                loss = 0.1 * loss1 + loss2
            else:
                loss1 = self.model.compute_objective(coarse_logit, batch['query_label'])
                loss2 = self.model.compute_objective(refined_logit, batch['query_label'])
                loss = 0.1 * loss1 + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 3. Evaluate prediction
            pred = refined_logit.argmax(dim=1)
            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)
            pass

        # Write evaluation results
        average_meter.write_result('Training', epoch)
        avg_loss = MyCommon.mean(average_meter.loss_buf)
        miou, fb_iou = average_meter.compute_iou()
        return avg_loss, miou, fb_iou

    def test_one_epoch(self, epoch, dataloader):
        has_module = hasattr(self.model, "module")
        self.model.module.eval() if has_module else self.model.eval()
        average_meter = AverageMeter(dataloader.dataset, device=self.device)

        for idx, batch in enumerate(dataloader):
            # 1. forward pass
            batch = MyCommon.to_cuda(batch, device=self.device)
            coarse_logit, refined_logit  = self.model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_labels'].squeeze(1),
                               query_mask=batch['query_mask'] if 'query_mask' in batch else None,
                               support_masks=batch['support_masks'].squeeze(1) if 'support_masks' in batch else None)
            # 2. Compute loss & update model parameters
            if has_module:
                loss1 = self.model.module.compute_objective(coarse_logit, batch['query_label'])
                loss2 = self.model.module.compute_objective(refined_logit, batch['query_label'])
                loss = 0.1 * loss1 + loss2
            else:
                loss1 = self.model.compute_objective(coarse_logit, batch['query_label'])
                loss2 = self.model.compute_objective(refined_logit, batch['query_label'])
                loss = 0.1 * loss1 + loss2
            # 3. Evaluate prediction
            pred = refined_logit.argmax(dim=1)
            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)
            pass

        # Write evaluation results
        average_meter.write_result('Validation', epoch)
        avg_loss = MyCommon.mean(average_meter.loss_buf)
        # Per Class IoU
        # miou, fb_iou, iou_per_class, class_ids  = average_meter.compute_iou_class()
        # print(f"[Epoch {epoch}] Per-class IoU:")
        # for cid, iou in zip(class_ids, iou_per_class):
        #     print(f"  Class {cid}: {iou.item():.4f}")
        
        miou, fb_iou = average_meter.compute_iou()
        return avg_loss, miou, fb_iou

    pass


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <config.yaml>")
        exit(1)

    yaml_path = sys.argv[1]
    args = load_yaml_config(yaml_path)
    runner = Runner(args=args)
    runner.train()