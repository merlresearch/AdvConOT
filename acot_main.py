# Copyright (c) 2020,2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import print_function

import argparse
import random
import torch

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import acot_data_loader as hmdb
import acot_pooling as cot
import numpy as np
from sklearn import svm
from tqdm import tqdm
import os

import copy

import mlp_hmdb as mlp
from classify import classify, compute_accuracy, compute_accuracy_against_numframes

def print_perf(perf):
    classid, num_seq, seq_max, seq_min, seq_mean, class_acc, overall = perf

    for t in range(len(classid)):
        print('classid=%d mean_frames=%d seq_acc=%f' % (classid[t], seq_mean[t], class_acc[t]))
    print('overall accuracy=%f'%(overall))

def stat_perf(y_pred, l_test, num_frames):
    classes = np.unique(l_test)
    num_seq, max_frames, min_frames, mean_frames, acc = np.zeros((len(classes),)) , np.zeros((len(classes),)), np.zeros((len(classes),)), np.zeros((len(classes),)), np.zeros((len(classes),))
    classid = np.zeros((len(classes),))
    for t in range(len(classes)):
        idx =  np.where(l_test == classes[t])[0]
        classid[t] = classes[t]
        num_seq[t] = len(idx)
        seq_stats = num_frames[idx]
        max_frames[t] = seq_stats.max()
        min_frames[t] = seq_stats.min()
        mean_frames[t] = seq_stats.mean()
        acc[t] = (np.array(y_pred[idx])==classid[t]).sum()/float(len(idx))
    overall_acc = np.array(acc).sum()/float(len(acc))
    return (classid, num_seq, max_frames, min_frames, mean_frames, acc, overall_acc)

# submodule to solve the WGAN to generate adversarial noise distributions.
def solve_WGAN(opt):
    if opt.experiment is None:
        opt.experiment = 'samples'
    os.system('mkdir {0}'.format(opt.experiment))

    opt.manualSeed = 1234
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    dataloader = None

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.dataset == 'hmdb_i3d' or opt.dataset == 'hmdb_i3d_jue':
        num_classes = 51
        if opt.test == False:
            dataloader = hmdb.get_train_loader(opt.datatype, batch_size=opt.batchSize,
                                                    shuffle=True, num_workers=int(opt.workers), split_num=opt.split_num)
    else:
        raise SystemExit('Unknown dataset')

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = int(opt.nc)
    n_extra_layers = int(opt.n_extra_layers)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def data_merge(x, y):
        yy=[torch.tensor([y[t]]*len(x[t])) for t in range(len(y))]
        return torch.cat(x, dim=0).float(), torch.cat(yy, dim=0)

    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif opt.mlp_G:
        netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
    else:
        netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    netG.apply(weights_init)
    if opt.netG != '' and os.path.exists(opt.netG): # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))
    print('Generator')
    print(netG)

    netC = mlp.Classifier(nz, nc, num_classes, ndf, ngpu) # isize, nc, num_classes, ndf, ngpu
    print('classifier')
    print(netC)

    if opt.netC != '' and os.path.exists(opt.netC):
        netC.load_state_dict(torch.load(opt.netC))

    if opt.mlp_D:
        netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
    else:
        netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
        netD.apply(weights_init)

    if opt.netD != '' and os.path.exists(opt.netD): # != '':
        netD.load_state_dict(torch.load(opt.netD))
    print('Discriminator')
    print(netD)

    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        netC.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    # setup optimizer
    optimizerC = optim.Adam(netC.parameters(), lr=opt.lrC, betas=(opt.beta1, 0.999))
    if opt.adam:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

    if opt.test == False and ((opt.netC == '') or (opt.netC != '' and not os.path.exists(opt.netC))):
        # lets first train the classifier.
        print('training the classifier... for %s'%(opt.datatype))
        for p in netC.parameters():
            p.requires_grad = True

        for epoch in range(opt.cl_iter):
             data_iter = iter(dataloader)
             i = 0
             avg_classifier_loss, avg_pivot_ae_loss, avg_pivot_cl_loss, avg_acc = 0., 0., 0., 0.
             while i < len(dataloader):
                 i += 1
                 x, y = data_iter.next()
                 netC.zero_grad()

                 x, y = Variable(x), Variable(y)
                 if opt.cuda:
                     x, y = x.cuda(), y.cuda()

                 classifier_loss, y_pred = netC(x, y, one)
                 classifier_loss.backward()

                 optimizerC.step()

                 avg_classifier_loss += classifier_loss.data
                 avg_acc += float((y_pred.data.argmax(dim=1) == y).sum())/float(len(y))

             if (epoch+1) % 1 == 0:
                 n = len(dataloader)
                 print('%d: avg_pivot_cl_loss = %f avg_classifier_loss=%f avg_acc = %f ' %
                       (epoch+1, avg_pivot_cl_loss/n, avg_classifier_loss/n, avg_acc/n))


        torch.save(netC.state_dict(), '{0}/{1}_{2}_split{3}_netC_epoch_{4}.pth'.format(opt.experiment, opt.dataset, opt.datatype, opt.split_num, epoch))

    if opt.test == True: # that is, we need not train the model, but use the existing model if available.
        if opt.netG != '' and os.path.exists(opt.netG):
            return netG, None, netC
        else:
            return None, None, None

    print('Training GAN to generate advesrial noise distribution conditioned on the input data...')
    gen_iterations = 0
    avg_real_acc, avg_faked_G_acc_gen, avg_faked_G_norm, avg_fake, disc_count, gen_count = 0., 0., 0., 0., 0., 0.0
    for epoch in range(opt.niter):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update
            for p in netC.parameters():
                p.requires_grad = True

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                data = data_iter.next()
                i += 1

                # train with real
                real_cpu, gt_label = data
                netD.zero_grad()
                netC.zero_grad()
                batch_size = real_cpu.size(0)

                if opt.cuda:
                    real_cpu = real_cpu.cuda()
                    gt_label = gt_label.cuda()
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv, gt_label = Variable(input), Variable(gt_label)

                errD_real = netD(inputv)
                errD_real.backward(one)

                real_classifier_loss, real_pred = netC(inputv, gt_label, one)
                avg_real_acc += ((real_pred.data.argmax(dim=1) == gt_label).sum().float()/float(len(gt_label)))
                disc_count += 1

                # train with fake
                faked_input_D = inputv
                noise.resize_(faked_input_D.shape[0], nz, 1, 1).normal_(0., opt.train_sigma)
                shifted_noise_D = faked_input_D.data.unsqueeze(2).unsqueeze(2) + noise # shift to sample from N(pivot, I)
                with torch.no_grad():
                    shifted_noisev_D = Variable(shifted_noise_D) # totally freeze netG
                shifted_gen_noisev_D = Variable(netG(shifted_noisev_D).data)

                classifier_loss = real_classifier_loss

                faked_input_D = torch.relu(faked_input_D - shifted_gen_noisev_D)
                faked_input_D = faked_input_D/(torch.norm(faked_input_D,dim=1).unsqueeze(1) + 1e-10)
                errD_fake = netD(faked_input_D)
                errD_fake.backward(mone)
                classifier_loss.backward()

                errD = errD_real - errD_fake
                optimizerD.step()
                optimizerC.step()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            for p in netC.parameters():
                p.requires_grad = False # to avoid training the classifier when learning G.
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise

            faked_input_G = inputv
            noise.resize_(faked_input_G.shape[0], nz, 1, 1).normal_(0, opt.train_sigma)
            shifted_noise_G = faked_input_G.data.unsqueeze(2).unsqueeze(2) + noise # shift to sample from N(pivot, I)
            shifted_noisev_G = Variable(shifted_noise_G)
            shifted_gen_noisev_G = netG(shifted_noisev_G)
            faked_input_G = torch.relu(faked_input_G - shifted_gen_noisev_G)

            faked_input_G_norm = shifted_gen_noisev_G.norm(dim=1)
            faked_input_G_norm = (faked_input_G_norm * faked_input_G_norm).mean()
            faked_input_G_classifier_loss, faked_input_G_pred = netC(faked_input_G, gt_label, mone)

            xx = netC(inputv).data; yy=netC(faked_input_G).data;
            avg_fake += (float(torch.sum(xx.argmax(dim=1) == yy.argmin(dim=1)))/float(len(gt_label))) # xx.argmax(dim=1)

            avg_faked_G_acc_gen += (faked_input_G_pred.data.argmax(dim=1) == gt_label).sum().float()/float(len(gt_label)) #fake_classifier_loss.data #
            avg_faked_G_norm += faked_input_G_norm.data

            gen_count += 1

            errG = netD(faked_input_G) + faked_input_G_classifier_loss + opt.beta*faked_input_G_norm
            errG.backward(one)
            optimizerG.step()

            gen_iterations += 1

            echo_freq = 100
            if (epoch+1) % echo_freq == 0:
                print('argmax==argmin performance = %f: acc_fake_acc_gen=%f\n' % (avg_fake/gen_iterations, avg_faked_G_acc_gen/gen_count))
                print('[%d/%d][%d/%d][%d] L_D: %0.3f L_G: %0.3f L_D_real: %0.3f L_D_fake %0.3f Acc_real: %f Acc_fake_gen: %0.3f F_Norm: %f  n_min: %0.3f n_max: %0.3f '
                    % (epoch+1, opt.niter, i, len(dataloader), gen_iterations,
                    errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0],
                    avg_real_acc/disc_count, avg_faked_G_acc_gen/gen_count, avg_faked_G_norm/gen_count,
                     noise.data.min(), noise.data.max()))
                avg_real_acc,avg_faked_G_acc_gen, avg_faked_G_norm, disc_count, gen_count, avg_fake = 0., 0., 0., 0., 0., 0.

        # do checkpointing
        if epoch == opt.niter or ((epoch+1) % 500 == 0):
            if opt.dataset not in mydatasets:
                torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch+1))
                torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch+1))
                torch.save(netC.state_dict(), '{0}/netC_epoch_{1}.pth'.format(opt.experiment, epoch+1))
            else:
                print('saving {0}/{1}_{2}_sigma_{3}_split{4}_netG_epoch_{5}.pth'.format(opt.experiment, opt.dataset, opt.datatype, opt.train_sigma, opt.split_num, epoch+1))
                torch.save(netG.state_dict(), '{0}/{1}_{2}_sigma_{3}_split{4}_netG_epoch_{5}.pth'.format(opt.experiment, opt.dataset, opt.datatype, opt.train_sigma, opt.split_num, epoch+1))
                torch.save(netD.state_dict(), '{0}/{1}_{2}_sigma_{3}_split{4}_netD_epoch_{5}.pth'.format(opt.experiment, opt.dataset, opt.datatype, opt.train_sigma, opt.split_num, epoch+1))
                torch.save(netC.state_dict(), '{0}/{1}_{2}_sigma_{3}_split{4}_netC_epoch_{5}.pth'.format(opt.experiment, opt.dataset, opt.datatype, opt.train_sigma, opt.split_num, epoch+1))
    return netG, None, netC

class WGAN_COT():
    def __init__(self, wgan_model, ae_model, cl_model, num_samples=100, sigma=0.01, nz=100):
        self.netG = wgan_model
        self.netA = ae_model
        self.nz = nz
        self.sigma = sigma
        self.num_samples = num_samples
        self.classifier = cl_model
        self.noise = torch.FloatTensor(self.num_samples, self.nz).normal_(0, self.sigma)

    def replicate(self, X, num_samples):
        """
        randomly replicate the rows of X to match num_samples.
        """
        return X.repeat(num_samples,1) # we repeat X num_sample times.

    def generate_negative_distr(self, X):
        Y = []
        if self.num_samples > 1: # number of replicates for a given data sample.
            pivot = self.replicate(X, self.num_samples)
        else:
            pivot = X
        with torch.no_grad():
            inputv = Variable(pivot).cuda().float()
            self.noise.resize_(inputv.shape[0], self.nz).normal_(0, self.sigma)
            noisev = inputv + self.noise.cuda()
            fake = self.netG(noisev)
            fakev = torch.relu(inputv - fake)
            fakev = fakev/(torch.norm(fakev, dim=1) + 1e-10).unsqueeze(1)
            Y.append(fakev.cpu().data)

        return Y


def solve_metric_OT(wgan_adv_model, ae_encoder_model, cl_model, data_type, num_samples=100, sigma=0.01,
                    nz=100, num_subspaces=1, eta=0.01, lambda_val=1.,pca_val=1.0,
                    max_iter=1, split_num=1, no_OT=False, num_iter=5, two_way=False, skip=0, max_frames=30):
    wgan_cot = WGAN_COT(wgan_adv_model, ae_encoder_model, cl_model, num_samples, sigma, nz)
    dataloader = hmdb.get_all_data_loader(data_type, batch_size=1, shuffle=False, num_workers=4, split_num=split_num)
    desc = []

    print('generating subspaces ...')
    for index, data in tqdm(enumerate(dataloader)):
        X, label, X_mean,_ = data[0], data[1], data[2], data[3]
        if X[0].shape[0]>max_frames:
            ii = np.sort(np.unique(np.linspace(0, X[0].shape[0]-1, max_frames).astype(int)))
            X[0] = X[0][ii,:]
        if X[0].shape[0]>10: # skip the last frame. They seem to be noisy for hmdb.
            X[0] = X[0][:-1,:]

        if wgan_adv_model is None:
            Y = None
        else:
            X_mean = X[0]#.mean(0)[np.newaxis,:].repeat(X[0].shape[0],1) # we could use mean. seems using the features directly seem better for hmdb.
            Y = wgan_cot.generate_negative_distr(X_mean)
            Y = torch.cat(Y, 0).data.cpu().numpy().transpose()
            Y=Y/np.linalg.norm(Y,axis=0)[np.newaxis,:]
        X = torch.cat(X, 0).data.cpu().numpy().transpose()
        if X.shape[1] == 1: # don't do any pooling if there is only one feature for the "sequence"
             U = X;
        else:
             U, _, Pi = cot.meta_solver(X, Y, p=num_subspaces, eta=eta, lambda_val=lambda_val, pca_val=pca_val, max_iter=max_iter, use_OT=not no_OT, num_iter=num_iter)
        if two_way == True: # we could use two way reasoning. however this seems not so helpful on hmdb.
            U_rev, _, Pi = cot.meta_solver(X[:,::-1], Y, p=num_subspaces, eta=eta, lambda_val=lambda_val, pca_val=pca_val, max_iter=max_iter, use_OT=not no_OT, num_iter=num_iter)
            U = np.concatenate([U,U_rev], axis=1).reshape(-1,1)
        desc.append(U)
    return desc

# main function ....
if __name__ == "__main__":
    np.random.seed(1234)
    print("Main: Random Seed: ", 1234)
    random.seed(1234)
    torch.manual_seed(1234)
    np.random.RandomState(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hmdb_i3d', help='cifar10 | lsun | imagenet | folder | lfw | hmdb')
    parser.add_argument('--dataroot', type=str, default='', help='path to dataset')
    parser.add_argument('--datatype', type=str, default='', help='rgb | flow')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=1, help='input image channels')
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=5000, help='number of epochs to train WGAN for')
    parser.add_argument('--aeiter', type=int, default=1000, help='number of epochs to train AE for')
    parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate for Critic, default=0.0001')
    parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate for Generator, default=0.0001')
    parser.add_argument('--lrA', type=float, default=0.0001, help='learning rate for AutoEncoder, default=0.0001') # default=0.00005
    parser.add_argument('--lrC', type=float, default=0.0001, help='learning rate for Real Classifier, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--netC', default='', help="path to netC (to continue training)")
    parser.add_argument('--netA', default='', help="path to netA (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
    parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default='./samples/retrained_i3d_from_jue/', help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--beta', type=float, default=1., help='threshold for the strength of the adversarial noise')
    parser.add_argument('--train_sigma', type=float, default=1.0, help='variance for the pivot noise')
    parser.add_argument('--split_num', type=int, default=1, help='data split number if there is one.')
    parser.add_argument('--eta', type=float, default=0.01, help='threshold for rank pooling')
    parser.add_argument('--lam', type=float, default=0.1, help='weight for rank pooling')
    parser.add_argument('--pca', type=float, default=1, help='weight for grp')
    parser.add_argument('--ot_iter', type=int, default=1, help='number of OT iterations')
    parser.add_argument('--num_subspaces', type=int, default=1, help='number of subspaces')
    parser.add_argument('--num_negatives', type=int, default=40, help='number of negative samples')
    parser.add_argument('--test_sigma', type=float, default=0.1, help='noise variance at inference time')
    parser.add_argument('--test', action='store_true', help='if true, then wgan training will not be done.')
    parser.add_argument('--flow_test', action='store_true', help='if true, then wgan training will not be done.')
    parser.add_argument('--rgb_test', action='store_true', help='if true, then wgan training will not be done.')
    parser.add_argument('--cl_iter', type=int, default=1000, help='number of classifier training iterations')
    parser.add_argument('--rgb_train', action='store_true', help='if true, then wgan training will not be done.')
    parser.add_argument('--flow_train', action='store_true', help='if true, then wgan training will not be done.')
    parser.add_argument('--no_OT', action='store_true', help='if true, do not use OT in the optimization')
    parser.add_argument('--no_adv', action='store_true', help='do not use wgan')
    parser.add_argument('--num_iter', type=int, default=5, help='number of iterations in CG')
    parser.add_argument('--two_way', action='store_true', help='should use bidirectional pooling')
    parser.add_argument('--skip', type=int, default=0, help='number of frames to temporally skip')
    parser.add_argument('--max_frames', type=int, default=100, help='number of max frames to use')

    opt = parser.parse_args()
    print(opt)

    split_num = opt.split_num

    rank_eta = opt.eta
    rank_lambda = opt.lam
    pca_val = opt.pca
    max_iter= opt.ot_iter
    num_subspaces = opt.num_subspaces
    num_negatives = opt.num_negatives
    num_iter = opt.num_iter
    nz = opt.nz
    OT_model_type = 'PCA+U+Slack'

    opt_rgb = copy.deepcopy(opt)
    opt_rgb.datatype = 'rgb'
    if (opt.rgb_test == True or opt.test == True):
        opt_rgb.test = True
    opt_flow = copy.deepcopy(opt)
    opt_flow.datatype = 'flow'
    if opt.flow_test == True or opt.test == True:
        opt_flow.test = True

    # in case pretrained models are not supplied, we will use the best ones.
    if opt_rgb.netG == '':
        opt_rgb.netC =  '{0}/{1}_{2}_split{3}_netC_epoch_{4}.pth'.format(opt.experiment, opt.dataset, opt_rgb.datatype, opt.split_num, opt.cl_iter-1)
        opt_rgb.netG =  '{0}/{1}_{2}_sigma_{3}_split{4}_netG_epoch_{5}.pth'.format(opt.experiment, opt.dataset, opt_rgb.datatype, opt.train_sigma, opt.split_num, opt.niter)
        opt_rgb.netD =  '{0}/{1}_{2}_sigma_{3}_split{4}_netD_epoch_{5}.pth'.format(opt.experiment, opt.dataset, opt_rgb.datatype, opt.train_sigma, opt.split_num, opt.niter)


    if opt_flow.netG == '':
        opt_flow.netC = '{0}/{1}_{2}_split{3}_netC_epoch_{4}.pth'.format(opt.experiment, opt.dataset, opt_flow.datatype, opt.split_num, opt.cl_iter-1)
        opt_flow.netG = '{0}/{1}_{2}_sigma_{3}_split{4}_netG_epoch_{5}.pth'.format(opt.experiment, opt.dataset, opt_flow.datatype, opt.train_sigma, opt.split_num, opt.niter)
        opt_flow.netD = '{0}/{1}_{2}_sigma_{3}_split{4}_netD_epoch_{5}.pth'.format(opt.experiment, opt.dataset, opt_flow.datatype, opt.train_sigma, opt.split_num, opt.niter)

    niter = opt.niter
    feat_rgb = 'hmdb_clean_i3d_flow_desc_split' + str(split_num) + OT_model_type + '_sub' + str(num_subspaces)+'_eta' + str(rank_eta) \
                 + '.pkl'
    feat_flow = 'hmdb_clean_i3d_rgb_desc_split' + str(split_num) + OT_model_type + '_sub' + str(num_subspaces)+'_eta' + str(rank_eta) \
                 + '.pkl'

    rgb_wgan_adv_model, rgb_ae_encoder_model, rgb_cl_model = None, None, None
    flow_wgan_adv_model, flow_ae_encoder_model, flow_cl_model = None, None, None

    if opt.rgb_train == False and opt.flow_train == False:
        print('rgb train or flow train unspecified and the mode is not test. So training for both rgb and flow streams.')
        opt.rgb_train = True
        opt.flow_train = True

    if opt.rgb_train or opt.test == True:
        print('training for rgb stream...')
        if opt.no_adv == False:# do not use adv pertubations.
            rgb_wgan_adv_model, rgb_ae_encoder_model, rgb_cl_model = solve_WGAN(opt_rgb)

    if opt.flow_train or opt.test == True:
        print('training for flow stream...')
        if opt.no_adv == False:# do not use adv pertubations.
            flow_wgan_adv_model, flow_ae_encoder_model, flow_cl_model = solve_WGAN(opt_flow)

    print('encoding... sigma = %f\n' %(opt.test_sigma))

    x_rgb, x_flow, labels, train_idx, test_idx, num_frames = hmdb.load_data_myi3d_trained_on_jues_feat(split_num=split_num, datatype='info')

    if rgb_wgan_adv_model == None:
        print('RGB: rgb_wgan_adv_model is None!!! ')

    # train or load pre-trained rgb model.
    X_rgb = solve_metric_OT(rgb_wgan_adv_model, rgb_ae_encoder_model, rgb_cl_model, 'rgb',
                               sigma=opt.test_sigma, nz=nz, num_subspaces=num_subspaces, num_samples=num_negatives,
                               eta=rank_eta, lambda_val=rank_lambda, pca_val = pca_val, max_iter=max_iter,
                               split_num=split_num, no_OT=opt.no_OT, num_iter=opt.num_iter, two_way=opt.two_way,skip=opt.skip, max_frames=opt.max_frames)
    if flow_wgan_adv_model == None:
        print('FLOW: flow_wgan_adv_model is None!!! ')

    # train or load pre-train flow model.
    X_flow = solve_metric_OT(flow_wgan_adv_model, flow_ae_encoder_model, flow_cl_model, 'flow',
                                sigma=opt.test_sigma, nz=nz, num_subspaces=num_subspaces, num_samples=num_negatives,
                                eta=rank_eta, lambda_val=rank_lambda, pca_val = pca_val, max_iter=max_iter,
                                split_num=split_num, no_OT=opt.no_OT, num_iter=opt.num_iter, two_way=opt.two_way, skip=opt.skip, max_frames=opt.max_frames)

    # load the split train test indicies.
    _, _, labels, train_idx, test_idx, num_frames = hmdb.load_data_i3d(split_num=split_num)

    # concatenate rgb and flow features
    X_train = np.concatenate([np.concatenate([X_rgb[i], X_flow[i]], axis=0).reshape(-1)[:,np.newaxis] for i in train_idx], axis=1).transpose();
    l_train = [labels[i] for i in train_idx]
    X_test = np.concatenate([np.concatenate([X_rgb[i], X_flow[i]], axis=0).reshape(-1)[:,np.newaxis] for i in test_idx], axis=1).transpose();
    l_test = [labels[i] for i in test_idx]

    # train and test the RGB || FLOW features.
    C = 1.
    classifier = svm.LinearSVC(C=C)
    classifier.fit(X_train, l_train)
    y_pred = classifier.predict(X_train)
    acc_train = compute_accuracy(y_pred, l_train)
    y_pred = classifier.predict(X_test)
    acc =  compute_accuracy(y_pred, l_test)
    print('RGB + FLOW: train_accuracy = %f test_accuracy = %f\n'%(acc_train, acc))
