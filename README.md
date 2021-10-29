# Detecting Invisible People

\[[ICCV 2021 Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Khurana_Detecting_Invisible_People_ICCV_2021_paper.html)\] \[[Website](http://www.cs.cmu.edu/~tkhurana/invisible.htm)\]

[Tarasha Khurana](http://www.cs.cmu.edu/~tkhurana/), [Achal Dave](http://www.achaldave.com), [Deva Ramanan](http://www.cs.cmu.edu/~deva/)

## Introduction

This repository contains code for *Detecting Invisible People*.
We extend the original [DeepSORT](https://github.com/nwojke/deep_sort) algorithm to
localize people even while they are completely occluded in a video.
See the [arXiv preprint](https://arxiv.org/abs/2012.08419) for more information.

## Dependencies

Create a conda environment with the given `environment.yml` file.

```
conda env create -f environment.yml
```

## Preprocessing

The code expects the directory structure of your dataset in the MOT Challenge
data format, which is approximately like the following:

```
MOT17/
-- train/
---- seq_01/
------ img1/
------ img1Depth/
------ gt/
------ det/
...
-- test/
---- seq_02/
------ img1/
------ img1Depth/
------ det/
```

The folder `img1Depth` stores the **normalized disparity** in `.npy` format. See
[Note](https://github.com/tarashakhurana/detecting-invisible-people#note). Originally, the paper runs
the method on depth given by the [MegaDepth](https://github.com/zl548/MegaDepth) depth estimator.

Given the above folder structure, generate the appearance features for your detections as
described in the DeepSORT [repository](https://github.com/nwojke/deep_sort#generating-detections).

## Running the method

The script `run_forecast_filtering.sh` will run the method with hyperparameters used in the paper.
It will produce output `.txt` files in the MOT Challenge submission format. The bashscript has support
for computing the metrics, but this has not been verified. Run the bashscript like the following:

```
bash run_forecast_filtering.sh experimentName
```

Note that in order to speed up code release, the dataset, preprocessed detections and output file paths
are hardcoded in the files and will have to be manually changed.

## Citing Detecting Invisible People

If you find this code useful in your research, please consider citing the following paper:

    @inproceedings{khurana2021detecting,
      title={{Detecting Invisible People}},
      author={Khurana, Tarasha and Dave, Achal and Ramanan, Deva},
      booktitle={{IEEE/CVF International Conference on Computer Vision (ICCV)}},
      year={2021}
    }

## Warning

This is only the starter code that has not been cleaned for release.
It currently only has verified support for running the method described in
*Detecting Invisible People*, with the output tracks written in the MOT Challenge
submission format. Although Top-k metric's code has been provided, this codebase
does not guarantee support for the metric yet.

The hope is that you are able to benchmark this method for your CVPR 2022 submission
and compute your own metrics on the method's output. If the method code does not work,
please open an issue.

## Note

Although it is easy to run any monocular depth estimator and store their output (usually given as
 disparity) in an `.npy` file, I have added a script in `tools/demo_images.py` which can save the `.npy`
files for you. Note that this script should be run after setting up the [MegaDepth](https://github.com/zl548/MegaDepth)
codebase and copying this file to its root directory. I will likely also release my own depth maps for the MOT17 dataset
over the Halloween weekend.

If you try to run the metrics, I have given my [groundtruth JSON](./evaluation) (as expected by `pycocotools`).
