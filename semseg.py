#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-15
#
# MES rev:  April 2019 to semantically segment images for use with MegaDepth depth estimation
#
# import click
import cv2
# MES change to deal with headless
import matplotlib as mpl
mpl.use('Agg')
#
import matplotlib.pyplot as plt
# MES change to use GPU 1
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from torch.autograd import Variable

from libs.models import PSPNet
from libs.utils import dense_crf

"""
@click.command()
@click.option("--config", "-c", required=True)
@click.option("--image-path", "-i", required=True)
@click.option("--cuda/--no-cuda", default=True)
@click.option("--crf", is_flag=True)
"""
#
#  Example parameters:
#  config = '../pspnet-pytorch/config/ade20k.yaml'
#  image_path = '../MegaDepth/docs/vertical-street.jpg'
#  cuda = True
#  crf = True
#  out_class_figure = 'docs/demo_out.png'
#  out_masked_sky_image = 'docs/image-sky-masked.png'
#
def semseg(config, image_path, cuda, crf):
    CONFIG = Dict(yaml.load(open(config)))

    cuda = cuda and torch.cuda.is_available()

    # Label list
    with open(CONFIG.LABELS) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split("\t")
            classes[int(label[0])] = label[1].split(",")[0]

    # Load a model
    state_dict = torch.load(CONFIG.PYTORCH_MODEL)

    # Model
    model = PSPNet(
        n_classes=CONFIG.N_CLASSES, n_blocks=CONFIG.N_BLOCKS, pyramids=CONFIG.PYRAMIDS
    )
    model.load_state_dict(state_dict)
    model.eval()
    if cuda:
        model.cuda()

    image_size = (CONFIG.IMAGE.SIZE.TEST,) * 2

    # Image preprocessing
    #  MES change for grayscale
    #  for color image input, use cv2.IMREAD_COLOR
    #  for grayscale image input, use cv2.IMREAD_GRAYSCALE
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(float)
    image = cv2.resize(image, image_size)
    image_original = image.astype(np.uint8)
    image = image[..., ::-1] - np.array(
        [CONFIG.IMAGE.MEAN.R, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.B]
    )
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.cuda() if cuda else image

    # Inference
    #  MES change to silence 'volatile' deprecation message
    with torch.no_grad():
        # output = model(Variable(image, volatile=True))
        output = model(Variable(image))

        # MES change to silence 'upsample' deprecation message
        # MES change to silence 'align_corners' default message
        # output = F.upsample(output, size=image_size, mode="bilinear")
        output = F.interpolate(output, size=image_size, mode="bilinear", align_corners=False)
        output = F.softmax(output, dim=1)
        output = output[0].cpu().data.numpy()

        if crf:
            output = dense_crf(image_original, output)
        #
    #
    labelmap = np.argmax(output.transpose(1, 2, 0), axis=2)

    labels = np.unique(labelmap)

    """
    rows = np.floor(np.sqrt(len(labels) + 1))
    cols = np.ceil((len(labels) + 1) / rows)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(rows, cols, 1)
    ax.set_title("Input image")
    ax.imshow(image_original[:, :, ::-1])
    ax.set_xticks([])
    ax.set_yticks([])
    """

    for i, label in enumerate(labels):
        print("{0:3d}: {1}".format(label, classes[label]))
        mask = labelmap == label
        """
        ax = plt.subplot(rows, cols, i + 2)
        ax.set_title(classes[label])
        ax.imshow(image_original[:, :, ::-1])
        ax.imshow(mask.astype(np.float32), alpha=0.5, cmap="viridis")
        ax.set_xticks([])
        ax.set_yticks([])
        """
        # MES change to save the sky class image as a separate image
        if classes[label] == 'sky':
            mask_invert = labelmap != label     # preserve non-sky pixels
            masked_image = cv2.bitwise_and(image_original, image_original, mask=mask_invert.astype(np.uint8))
            # plt.imsave(out_masked_sky_image, masked_image)

    # plt.tight_layout()
    # MES changes to save output
    # plt.show()
    # plt.savefig(out_class_figure)
    #
    # return the masked sky image
    return masked_image


# if __name__ == "__main__":
#     main()
