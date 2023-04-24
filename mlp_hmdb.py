# Copyright (c) 2020,2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn

class MLP_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu):
        super(MLP_G, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            nn.Linear(nz, nz),
            nn.ReLU(True),
            nn.Linear(nz, nz),
            nn.ReLU(True),
            nn.Linear(nz, nz),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0), input.size(1))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.size(0), self.nc*self.nz)


class MLP_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu):
        super(MLP_D, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            nn.Linear(nc * nz, nz),
            nn.ReLU(True),
            nn.Linear(nz, nz),
            nn.ReLU(True),
            nn.Linear(nz, 1),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.mean(0)
        return output.view(1)


class Classifier(nn.Module):
    def __init__(self, nz, nc, num_classes, ndf, ngpu):
        """
        isize: batch size,
        nc : num_channels = 1
        num_classes: number of data classes = 21
        ndf = latent code size = 100?
        ngpu = number of gpus
        """
        super(Classifier, self).__init__()
        self.ngpu = ngpu

        classify = nn.Sequential(
            nn.Linear(nc * nz, num_classes)
        )

        self.classify = classify
        self.nc = nc
        self.isize = nz
        self.num_classes = num_classes
        self.MSE_criterion = nn.MSELoss()
        self.CE_criterion = nn.CrossEntropyLoss()

    def classifier_test(self, input):
        return nn.functional.softmax(self.classify(input), dim=1)

    def classifier(self, input, gt_labels, realfake_sign):# realfake_sign = +1 if real, -1 if fake.
        y_pred = realfake_sign * self.classify(input)
        loss = self.CE_criterion(y_pred, gt_labels)
        return loss, nn.functional.softmax(y_pred, dim=1)

    def forward(self, input, gt_labels=None, realfake_sign=None):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output, loss = nn.parallel.data_parallel(self.classifier, input, gt_labels, realfake_sign, range(self.ngpu))
        else:
            if gt_labels is None: # test time
                return self.classifier_test(input)
            else:
                loss = self.classifier(input, gt_labels, realfake_sign)
                return loss

        return None
