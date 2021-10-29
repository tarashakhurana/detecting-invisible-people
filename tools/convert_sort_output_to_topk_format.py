import json
import argparse
import glob
import cv2
import os
import numpy as np
import time
from PIL import Image
from scipy.stats import norm, chi2

COLORMAP = {i: (np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255))
        for i in range(1,1000)}

def plot_cov_ellipse(cov, nstd=None, q=None):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations.
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    nsig = nstd
    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return height, width, rotation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--split", type=str,
            help="can be one of 'sort' or 'gt'", required=True)
    parser.add_argument("--interp", type=int, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--thresh", action="store_true")
    parser.add_argument("--k", type=int,
            help="how many samples to draw from the covariance")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--truncate", action="store_true")
    args = parser.parse_args()

    # ref = json.load(open('/data/all/coco/annotations/instances_val2017.json'))
    # tracks = sorted(glob.glob('../../../ECCV/deep_sort/results/outputminconf1e4MOT17/*FRCNN.*'))
    #tracks = sorted(glob.glob('/data/tkhurana/MOT17/train/*FRCNN/gt_interp/gt_interp.txt'))
    # tracks = sorted(glob.glob('/data/tkhurana/MOT17/train/*FRCNN/gt/gt_oracle_links_mot17det_trainall_cls127_occl05_interp.txt'))
    tracks = sorted(glob.glob(args.input))
    print(args.input)
    # tracks = sorted(glob.glob('/data2/tkhurana/DIP_EVAL/train/*/det/detCTrack.txt'))
    # data = sorted(glob.glob('/data/tkhurana/MOT17/train/*DPM*/img1/*.*'))
    data = sorted(glob.glob('/data2/tkhurana/DIP_EVAL/train/*/img1/*.*'))
    # basepath = '/data/tkhurana/MOT17/train/'
    basepath = '/data2/tkhurana/DIP_EVAL/train/'

    image_to_id = {}
    frame_to_img_id = {}

    for i, d in enumerate(data):
        # if '01_' not in d  and '02_' not in d and '03_' not in d:
        #     continue
        # if '-02-' not in d and '-04-' not in d and '-09-' not in d:
        #     continue
        if "Huaqiangbei" in d:
            continue
        img = Image.open(d)
        shape = img.size
        frame = int(d.split('/')[-1][:-4])
        seq = d.split('/')[5]
        depth_map = np.load(d.replace("SDP", "FRCNN").replace("DPM", "FRCNN").replace('/img1/', '/img1Depth/')[:-4] + '.npy')
        img_name = d[len(basepath):]
        if seq not in frame_to_img_id:
            frame_to_img_id[seq] = []
        image_to_id[img_name] = i
        frame_to_img_id[seq].append({'frame_id': frame, 'id': i, 'image': img_name, 'shape': shape, 'depth_map': depth_map})

    result = []

    widths, heights, box_heights = [], [], []

    print("done")
    for tr in tracks:
        # print("1")
        # if '-02-' not in tr and '-04-' not in tr and '-09-' not in tr:
        #     continue
        # if '-05-' not in tr and '-10-' not in tr and '-11-' not in tr and '-13-' not in tr:
        #     continue
        if "Huaqiangbei" in tr:
            continue
        # if '01_' not in tr  and '02_' not in tr and '03_' not in tr:
        #     continue
        if 'eval' in tr:
            continue
        print(args.interp, tr)
        if args.interp == 1 and "_interp.txt" not in tr:
            continue
        if args.split == 'sort':
            seq = tr.split('/')[-1][:-4]
        elif args.split == 'gt':
            seq = tr.split('/')[5]
        if args.interp == 1:
            seq = seq[:seq.rfind('_')]
        print(seq)
        # seq = tr.split('/')[5]
        lines = open(tr).readlines()
        for dssd, line in enumerate(lines):
            # print("Doing", dssd, len(lines))
            start = time.time()
            i = 0
            fields = line.strip().split(',')[:7]
            covariance = line.strip().split(',')[-4:]
            frame_id, track_id, x0, y0, w, h, score = fields
            xx, xz, zx, zz = covariance
            frame_id = int(frame_id)
            # if frame_id not in range(1,3500,15):
            correct_frame = [f for f in frame_to_img_id[seq] if f['frame_id'] == frame_id][0]
            #     continue
            track_id = int(track_id)
            x0 = float(x0)
            y0 = float(y0)
            w = float(w)
            h = float(h)
            score = float(score)
            depth = score
            depth_map = correct_frame["depth_map"]
            x1 = x0 + w
            y1 = y0 + h

            topk = []
            topk.append([x0, y0, w, h])

            # use correct_frame instead of image
            z0 = int(depth * correct_frame["shape"][1])
            center_coordinates = (int((x0 + x1) / 2), z0)
            cov = np.array([[xx, xz], [zx, zz]], dtype='float')
            if float(xx) < 0 or float(zz) < 0:
                continue
            # print(cov)
            width, height, angle = plot_cov_ellipse(cov, nstd=1)
            # print(width, height)
            if width <= 0 or height <= 0:
                continue
            width = int(width/2)
            height = int(correct_frame["shape"][1]*height/2)

            widths.append(width)
            heights.append(height)
            box_heights.append(h)

            k = 1
            #print("time 1", time.time() - start)

            start = time.time()
            while k <= args.k - 1:
                i += 1
                if i == 500:
                    break
                pointx, pointz = np.random.normal(center_coordinates, [width, height])
                pointx, pointz = np.round([pointx, pointz], 2)
                # print("pointx pointz", pointx, pointz)
                if pointx < 0 or pointz < 0:
                    continue
                scalex = correct_frame["shape"][0] / depth_map.shape[1]
                scaley = correct_frame["shape"][1] / depth_map.shape[0]
                pointx_ = int(pointx / scalex)
                y0_ = int(y0 / scaley)
                w_ = int(w / scalex)
                h_ = int(h / scaley)

                pointx_ = min(pointx_, depth_map.shape[1]-1)

                apparent_depth = correct_frame["shape"][1] * np.mean(depth_map[y0_:y0_+h_, pointx_ - int(w_ / 2):pointx_ + int(w_ / 2)])

                if pointz < apparent_depth and args.truncate: # pointz
                    topk.append([pointx - int(w / 2), y0, w, h])
                    k += 1
                elif not args.truncate:
                    topk.append([pointx - int(w / 2), y0, w, h])
                    k += 1

                # apparent_depth = correct_frame["shape"][1] * depth_map[y0_, pointx_]

                # if pointz < apparent_depth and args.truncate: # pointz
                #     topk.append([pointx, y0, w, h])
                #     k += 1
                # elif not args.truncate:
                #     topk.append([pointx, y0, w, h])
                #     k += 1

            while len(topk) < args.k:
                topk.append([x0, y0, w, h])

            # print("time 2", time.time() - start)
            # print(i)
            if args.debug:
                path = os.path.join(basepath, correct_frame["image"])
                impath = os.path.join("/data/tkhurana/MOT17/topk_truncate_{}_debug".format(args.truncate), correct_frame["image"])
                if os.path.exists(impath):
                    debugimage = cv2.imread(impath)
                else:
                    debugimage = cv2.imread(path)
                for box in topk:
                    cv2.rectangle(debugimage, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), COLORMAP[track_id % 1000], 2)
                if not os.path.exists(os.path.dirname(impath)):
                    os.makedirs(os.path.dirname(impath))
                cv2.imwrite(impath, debugimage)


            if score <= 0.0 and args.thresh:
                # print("skipping")
                continue
            if not args.thresh:
                score = 1.0
            # else:
            #     score = 1.0
            vis = float(line.strip().split(',')[-2])
            # print(score)
            # print(frame_id)
            correct_frame_id = correct_frame["id"]
            result.append({'image_id': correct_frame_id, 'category_id': 1, 'bbox': topk, 'score': score})
    # print(len(result))

    widths = np.array(widths)
    heights = np.array(heights)

    print("mean median widths", np.mean(widths), np.median(widths))
    print("mean median heights", np.mean(heights), np.median(heights))
    print("mean median boxheights", np.mean(box_heights), np.median(box_heights))

    with open(args.output_json, 'w') as f:
        json.dump(result, f)

