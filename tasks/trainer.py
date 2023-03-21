import importlib
import os
import subprocess

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.hparams import hparams, set_hparams
import numpy as np
from utils.utils import plot_img, move_to_cuda, load_checkpoint, save_checkpoint, tensors_to_scalars, load_ckpt, Measure


class Trainer:
    def __init__(self):
        self.logger = self.build_tensorboard(save_dir=hparams['work_dir'], name='tb_logs')
        self.measure = Measure()
        self.dataset_cls = None
        self.metric_keys = ['psnr', 'ssim', 'lpips', 'lr_psnr']
        self.work_dir = hparams['work_dir']
        self.first_val = True

    def build_tensorboard(self, save_dir, name, **kwargs):
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir, **kwargs)

    def build_train_dataloader(self):
        dataset = self.dataset_cls('train')
        return torch.utils.data.DataLoader(
            dataset, batch_size=hparams['batch_size'], shuffle=True,
            pin_memory=False, num_workers=hparams['num_workers'])

    def build_val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_cls('valid'), batch_size=hparams['eval_batch_size'], shuffle=False, pin_memory=False)

    def build_test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_cls('test'), batch_size=hparams['eval_batch_size'], shuffle=False, pin_memory=False)

    def build_model(self):
        raise NotImplementedError

    def sample_and_test(self, sample):
        raise NotImplementedError

    def build_optimizer(self, model):
        raise NotImplementedError

    def build_scheduler(self, optimizer):
        raise NotImplementedError

    def training_step(self, batch):
        raise NotImplementedError

    def train(self):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        self.global_step = training_step = load_checkpoint(model, optimizer, hparams['work_dir'])
        self.scheduler = scheduler = self.build_scheduler(optimizer)
        scheduler.step(training_step)
        dataloader = self.build_train_dataloader()

        train_pbar = tqdm(dataloader, initial=training_step, total=float('inf'),
                          dynamic_ncols=True, unit='step')
        while self.global_step < hparams['max_updates']:
            for batch in train_pbar:
                if training_step % hparams['val_check_interval'] == 0:
                    with torch.no_grad():
                        model.eval()
                        self.validate(training_step)
                    save_checkpoint(model, optimizer, self.work_dir, training_step, hparams['num_ckpt_keep'])
                model.train()
                batch = move_to_cuda(batch)
                losses, total_loss = self.training_step(batch)
                optimizer.zero_grad()

                total_loss.backward()
                optimizer.step()
                training_step += 1
                scheduler.step(training_step)
                self.global_step = training_step
                if training_step % 100 == 0:
                    self.log_metrics({f'tr/{k}': v for k, v in losses.items()}, training_step)
                train_pbar.set_postfix(**tensors_to_scalars(losses))

    def validate(self, training_step):
        val_dataloader = self.build_val_dataloader()
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for batch_idx, batch in pbar:
            if self.first_val and batch_idx > hparams['num_sanity_val_steps']:  # 每次运行的第一次validation只跑一小部分数据，来验证代码能否跑通
                break
            batch = move_to_cuda(batch)
            img, rrdb_out, ret = self.sample_and_test(batch)
            img_hr = batch['img_hr']
            img_lr = batch['img_lr']
            img_lr_up = batch['img_lr_up']
            if img is not None:
                self.logger.add_image(f'Pred_{batch_idx}', plot_img(img[0]), self.global_step)
                if hparams.get('aux_l1_loss'):
                    self.logger.add_image(f'rrdb_out_{batch_idx}', plot_img(rrdb_out[0]), self.global_step)
                if self.global_step <= hparams['val_check_interval']:
                    self.logger.add_image(f'HR_{batch_idx}', plot_img(img_hr[0]), self.global_step)
                    self.logger.add_image(f'LR_{batch_idx}', plot_img(img_lr[0]), self.global_step)
                    self.logger.add_image(f'BL_{batch_idx}', plot_img(img_lr_up[0]), self.global_step)
            metrics = {}
            metrics.update({k: np.mean(ret[k]) for k in self.metric_keys})
            pbar.set_postfix(**tensors_to_scalars(metrics))
        if hparams['infer']:
            print('Val results:', metrics)
        else:
            if not self.first_val:
                self.log_metrics({f'val/{k}': v for k, v in metrics.items()}, training_step)
                print('Val results:', metrics)
            else:
                print('Sanity val results:', metrics)
        self.first_val = False

    def test(self):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        load_checkpoint(model, optimizer, hparams['work_dir'])
        optimizer = None

        self.results = {k: 0 for k in self.metric_keys}
        self.n_samples = 0
        self.gen_dir = f"{hparams['work_dir']}/results_{self.global_step}_{hparams['gen_dir_name']}"
        if hparams['test_save_png']:
            subprocess.check_call(f'rm -rf {self.gen_dir}', shell=True)
            os.makedirs(f'{self.gen_dir}/outputs', exist_ok=True)
            os.makedirs(f'{self.gen_dir}/SR', exist_ok=True)

        self.model.sample_tqdm = False
        torch.backends.cudnn.benchmark = False
        if hparams['test_save_png']:
            if hasattr(self.model.denoise_fn, 'make_generation_fast_'):
                self.model.denoise_fn.make_generation_fast_()
            os.makedirs(f'{self.gen_dir}/RRDB', exist_ok=True)
            os.makedirs(f'{self.gen_dir}/HR', exist_ok=True)
            os.makedirs(f'{self.gen_dir}/LR', exist_ok=True)
            os.makedirs(f'{self.gen_dir}/UP', exist_ok=True)

        with torch.no_grad():
            model.eval()
            test_dataloader = self.build_test_dataloader()
            pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
            for batch_idx, batch in pbar:
                move_to_cuda(batch)
                gen_dir = self.gen_dir
                item_names = batch['item_name']
                img_hr = batch['img_hr']
                img_lr = batch['img_lr']
                img_lr_up = batch['img_lr_up']

                if hparams['save_intermediate']:
                    item_name = item_names[0]
                    img, rrdb_out, imgs = self.model.sample(
                        img_lr, img_lr_up, img_hr.shape, save_intermediate=True)
                    os.makedirs(f"{gen_dir}/intermediate/{item_name}", exist_ok=True)
                    Image.fromarray(self.tensor2img(img_hr)[0]).save(f"{gen_dir}/intermediate/{item_name}/G.png")

                    for i, (m, x_recon) in enumerate(tqdm(imgs)):
                        if i % (hparams['timesteps'] // 20) == 0 or i == hparams['timesteps'] - 1:
                            t_batched = torch.stack([torch.tensor(i).to(img.device)] * img.shape[0])
                            x_t = self.model.q_sample(self.model.img2res(img_hr, img_lr_up), t=t_batched)
                            Image.fromarray(self.tensor2img(x_t)[0]).save(
                                f"{gen_dir}/intermediate/{item_name}/noise1_{i:03d}.png")
                            Image.fromarray(self.tensor2img(m)[0]).save(
                                f"{gen_dir}/intermediate/{item_name}/noise_{i:03d}.png")
                            Image.fromarray(self.tensor2img(x_recon)[0]).save(
                                f"{gen_dir}/intermediate/{item_name}/{i:03d}.png")
                    return {}

                res = self.sample_and_test(batch)
                if len(res) == 3:
                    img_sr, rrdb_out, ret = res
                else:
                    img_sr, ret = res
                    rrdb_out = img_sr
                img_hr = batch['img_hr']
                img_lr = batch['img_lr']
                img_lr_up = batch.get('img_lr_up', img_lr_up)
                if img_sr is not None:
                    metrics = list(self.metric_keys)
                    for k in metrics:
                        self.results[k] += ret[k]
                    self.n_samples += ret['n_samples']
                    print({k: round(self.results[k] / self.n_samples, 3) for k in metrics}, 'total:', self.n_samples)
                    if hparams['test_save_png'] and img_sr is not None:
                        img_sr = self.tensor2img(img_sr)
                        img_hr = self.tensor2img(img_hr)
                        img_lr = self.tensor2img(img_lr)
                        img_lr_up = self.tensor2img(img_lr_up)
                        rrdb_out = self.tensor2img(rrdb_out)
                        for item_name, hr_p, hr_g, lr, lr_up, rrdb_o in zip(
                                item_names, img_sr, img_hr, img_lr, img_lr_up, rrdb_out):
                            item_name = os.path.splitext(item_name)[0]
                            hr_p = Image.fromarray(hr_p)
                            hr_g = Image.fromarray(hr_g)
                            lr = Image.fromarray(lr)
                            lr_up = Image.fromarray(lr_up)
                            rrdb_o = Image.fromarray(rrdb_o)
                            hr_p.save(f"{gen_dir}/outputs/{item_name}[SR].png")
                            hr_g.save(f"{gen_dir}/outputs/{item_name}[HR].png")
                            lr.save(f"{gen_dir}/outputs/{item_name}[LR].png")
                            hr_p.save(f"{gen_dir}/SR/{item_name}.png")
                            hr_g.save(f"{gen_dir}/HR/{item_name}.png")
                            lr.save(f"{gen_dir}/LR/{item_name}.png")
                            lr_up.save(f"{gen_dir}/UP/{item_name}.png")
                            rrdb_o.save(f"{gen_dir}/RRDB/{item_name}.png")

    # utils
    def log_metrics(self, metrics, step):
        metrics = self.metrics_to_scalars(metrics)
        logger = self.logger
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    @staticmethod
    def tensor2img(img):
        img = np.round((img.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5)
        img = img.clip(min=0, max=255).astype(np.uint8)
        return img


if __name__ == '__main__':
    set_hparams()

    pkg = ".".join(hparams["trainer_cls"].split(".")[:-1])
    cls_name = hparams["trainer_cls"].split(".")[-1]
    trainer = getattr(importlib.import_module(pkg), cls_name)()
    if not hparams['infer']:
        trainer.train()
    else:
        trainer.test()
