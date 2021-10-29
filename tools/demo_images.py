import torch
import sys
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions
opt = TrainOptions().parse()
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize
import argparse
import json
import os
import time
import glob
import time
import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = 933120000


model = create_model(opt)

input_height = 384
input_width  = 512
batch_size = 1


def test_simple(model, images):
    all_times = []
    total_loss = 0
    toal_count = 0
    print("============================= TEST ============================")
    model.switch_to_eval()

    batch = []
    i = 0
    paths = []
    for img_path in images:
        # print('Processing', img_path)
        start = time.time()
        if i < batch_size:
            img = np.float32(io.imread(img_path))/255.0
            #input_height = img.shape[0]
            #input_width = img.shape[1]
            #print(img.shape)
            img = resize(img, (input_height, input_width), order = 1)

            #if input_height % 2 == 1:
            #    img = img[:input_height-1,:]
            #if input_width % 2 == 1:
            #    img = img[:,:input_width-1]

            #print(img.shape)
            #print(np.transpose(img, (2,0,1)).shape)
            paths.append(img_path)
            batch.append(np.transpose(img, (2,0,1)))
        else:
            batch = torch.from_numpy(np.array(batch)).contiguous().float()
            #print(img_ids)
            forward_pass(model, batch, paths)
            img = np.float32(io.imread(img_path))/255.0
            #input_height = img.shape[0]
            #input_width = img.shape[1]
            img = resize(img, (input_height, input_width), order = 1)

            #if input_height % 2 == 1:
            #    img = img[:input_height-1,:]
            #if input_width % 2 == 1:
            #    img = img[:,:input_width-1]

            batch = [np.transpose(img, (2,0,1))]
            paths = [img_path]
            i = 0
        i += 1
        per_frame = time.time() - start
        all_times.append(per_frame)

    print("Average time taken", np.mean(all_times))

    if len(batch) != 0:
        batch = torch.from_numpy(np.array(batch)).contiguous().float()
        #print(img_ids)
        forward_pass(model, batch, paths)


def forward_pass(model, input_img, paths):
    input_images = Variable(input_img.cuda() )
    with torch.no_grad():
        pred_log_depth = model.netG.forward(input_images)
    pred_log_depth = torch.squeeze(pred_log_depth)

    pred_depth = torch.exp(pred_log_depth)
    # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
    # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
    pred_inv_depth = 1/pred_depth
    pred_inv_depth = pred_inv_depth.data.cpu().numpy()
    pred_depth = pred_depth.data.cpu().numpy()

    # you might also use percentile for better visualization
    pred_inv_depth_normalised = pred_inv_depth/np.amax(pred_inv_depth)


    for idx, img in enumerate([pred_inv_depth_normalised]):
        fname = paths[idx].replace('/img1/', '/img1Depth/')
        if not os.path.exists(fname[:fname.rfind('/')]):
            os.makedirs(fname[:fname.rfind('/')])
        # print("Doing: ", fname)
        # io.imsave(fname, img)
        np.save(fname[:-4], img)

    """
    for idx, img in enumerate([pred_inv_depth]):
        fname = paths[idx].replace('/img1/', '/img1DepthUnnormalised/')
        if not os.path.exists(fname[:fname.rfind('/')]):
            os.makedirs(fname[:fname.rfind('/')])
        print("Doing: ", fname)
        #io.imsave(fname, img)
        np.save(fname[:-4], img)

    for idx, img in enumerate([pred_depth]):
        fname = paths[idx].replace('/img1/', '/img1DepthUninversedUnnormalised/')
        if not os.path.exists(fname[:fname.rfind('/')]):
            os.makedirs(fname[:fname.rfind('/')])
        print("Doing: ", fname)
        #io.imsave(fname, img)
        np.save(fname[:-4], img)
    """

if __name__ == "__main__":
    print('starting')
    images = sorted(glob.glob("/data/tkhurana/MOT20/test/*/img1/*.jpg"))
    print('got images')
    test_simple(model, images)
    print("We are done")
