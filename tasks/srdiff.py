import os.path

import torch
from models.diffsr_modules import Unet, RRDBNet
from models.diffusion import GaussianDiffusion
from tasks.trainer import Trainer
from utils.hparams import hparams
from utils.utils import load_ckpt


class SRDiffTrainer(Trainer):
    def build_model(self):
        hidden_size = hparams['hidden_size']
        dim_mults = hparams['unet_dim_mults']
        dim_mults = [int(x) for x in dim_mults.split('|')]
        denoise_fn = Unet(
            hidden_size, out_dim=3, cond_dim=hparams['rrdb_num_feat'], dim_mults=dim_mults)
        if hparams['use_rrdb']:
            rrdb = RRDBNet(3, 3, hparams['rrdb_num_feat'], hparams['rrdb_num_block'],
                           hparams['rrdb_num_feat'] // 2)
            if hparams['rrdb_ckpt'] != '' and os.path.exists(hparams['rrdb_ckpt']):
                load_ckpt(rrdb, hparams['rrdb_ckpt'])
        else:
            rrdb = None
        self.model = GaussianDiffusion(
            denoise_fn=denoise_fn,
            rrdb_net=rrdb,
            timesteps=hparams['timesteps'],
            loss_type=hparams['loss_type']
        )
        self.global_step = 0
        return self.model

    def sample_and_test(self, sample):
        ret = {k: 0 for k in self.metric_keys}
        ret['n_samples'] = 0
        img_hr = sample['img_hr']
        img_lr = sample['img_lr']
        img_lr_up = sample['img_lr_up']
        img_sr, rrdb_out = self.model.sample(img_lr, img_lr_up, img_hr.shape)
        for b in range(img_sr.shape[0]):
            s = self.measure.measure(img_sr[b], img_hr[b], img_lr[b], hparams['sr_scale'])
            ret['psnr'] += s['psnr']
            ret['ssim'] += s['ssim']
            ret['lpips'] += s['lpips']
            ret['lr_psnr'] += s['lr_psnr']
            ret['n_samples'] += 1
        return img_sr, rrdb_out, ret

    def build_optimizer(self, model):
        params = list(model.named_parameters())
        if not hparams['fix_rrdb']:
            params = [p for p in params if 'rrdb' not in p[0]]
        params = [p[1] for p in params]
        return torch.optim.Adam(params, lr=hparams['lr'])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def training_step(self, batch):
        img_hr = batch['img_hr']
        img_lr = batch['img_lr']
        img_lr_up = batch['img_lr_up']
        losses, _, _ = self.model(img_hr, img_lr, img_lr_up)
        total_loss = sum(losses.values())
        return losses, total_loss
