from munit.networks import AdaINGen, MsImageDis, VAEGen
from munit.utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
from model.deeplabv2 import get_deeplab_v2 #lys
from model.deeplabv2 import get_deeplab_v2  as get_deeplab_v2_bdl

class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters, opts):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        self.opts = opts
        self.device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

        # initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'], opts.gpu_ids)  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'], opts.gpu_ids)  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'], opts.gpu_ids)  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'], opts.gpu_ids)  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.instancenorm_cl = nn.InstanceNorm2d(opts.num_classes, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).to(self.device)
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).to(self.device)

        # setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # load semantic segmentation model (vgg)
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
            if len(opts.gpu_ids) > 0:
                assert (torch.cuda.is_available())
                self.vgg.to(opts.gpu_ids[0])
                self.vgg = torch.nn.DataParallel(self.vgg, opts.gpu_ids)
        
        # load semantic segmentation model (deeplab)
        if 'deeplab_w' in hyperparameters.keys() and hyperparameters['deeplab_w'] > 0:
            self.deeplab = get_deeplab_v2(num_classes=opts.num_classes, multi_level=False)
            deeplab_dict = self.deeplab.state_dict()
            pretrained_dict = torch.load(opts.deeplab_model_path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in deeplab_dict}
            self.deeplab.load_state_dict(pretrained_dict)
            self.deeplab.eval()
            for param in self.deeplab.parameters():
                param.requires_grad = False
            if len(opts.gpu_ids) > 0:
                assert (torch.cuda.is_available())
                self.deeplab.to(opts.gpu_ids[0])
                self.deeplab = torch.nn.DataParallel(self.deeplab, opts.gpu_ids)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

         # compute semantic features
        if hyperparameters['deeplab_w'] > 0:
            x_a_feat, x_b_feat = self.compute_semantic_feat(x_a, self.deeplab), self.compute_semantic_feat(x_b, self.deeplab)
            x_ab_feat, x_ba_feat = self.compute_semantic_feat(x_ab, self.deeplab), self.compute_semantic_feat(x_ba, self.deeplab)
        if hyperparameters['rec_x_deeplab_w'] > 0:
            x_aba_feat, x_bab_feat = self.compute_semantic_feat(x_aba, self.deeplab), self.compute_semantic_feat(x_bab, self.deeplab)

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        
        # semantic loss on the translated images
        self.loss_gen_deeplab_a = self.compute_deeplab_loss(x_ba_feat, x_b_feat) if hyperparameters['deeplab_w'] > 0 else 0
        self.loss_gen_deeplab_b = self.compute_deeplab_loss(x_ab_feat, x_a_feat) if hyperparameters['deeplab_w'] > 0 else 0
        
        # semantic loss on the reconstructed images
        self.loss_gen_deeplab_rec_a = self.compute_deeplab_loss(x_aba_feat, x_a_feat) if hyperparameters['rec_x_deeplab_w'] > 0 else 0
        self.loss_gen_deeplab_rec_b = self.compute_deeplab_loss(x_bab_feat, x_b_feat) if hyperparameters['rec_x_deeplab_w'] > 0 else 0

        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                              hyperparameters['deeplab_w'] * self.loss_gen_deeplab_a + \
                              hyperparameters['deeplab_w'] * self.loss_gen_deeplab_b + \
                              hyperparameters['rec_x_deeplab_w'] * self.loss_gen_deeplab_rec_a + \
                              hyperparameters['rec_x_deeplab_w'] * self.loss_gen_deeplab_rec_b
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def compute_semantic_feat(self, img, deeplab):
        img_vgg = vgg_preprocess(img)
        img_feat = deeplab(img_vgg)
        return img_feat[1]

    def compute_deeplab_loss(self, img_feat, target_feat):
        if self.opts.in_norm:
            return torch.mean((self.instancenorm_cl(img_feat) - self.instancenorm_cl(target_feat)) ** 2)
        else:
            return torch.mean((img_feat - target_feat) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))
        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations
    
    def resume_gen(self, model_name):
        # lload generators
        state_dict = torch.load(model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(model_name[-11:-3])
        print('Resume from iteration %d' % iterations)
        return iterations


    def save(self, snapshot_dir, iterations):
        # save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
