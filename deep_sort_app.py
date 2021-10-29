# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection, DetectionMask
from deep_sort.tracker import Tracker
import time


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file, allow_pickle=True)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def create_detections_mask(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        # print(row)
        bbox, confidence, mask, feature = row[2:6], row[6], ','.join(row[7:-128]), row[-128:]
        if bbox[3] < min_height:
            continue
        # print(mask)
        detection_list.append(DetectionMask(bbox, confidence, mask, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, depth_map_path, max_age, only_filtering,
        temporal_noise, default_matching, freespace_filtering,
        ah_velocity, velocity_weighting, tn, occluded_factor,
        filtering_factor, motion_aware, output_uncertainty,
        only_extrapolate, extrapolated_iou_match, appearance_match,
        bugfix):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=max_age)
    results = []

    def frame_callback(vis, frame_idx):
        startall = time.time()
        start = time.time()
        # print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        if len(detections) == 0:
            max_height = -1
        else:
            max_height = max(np.asarray([d.tlwh[3] for d in detections]))

        # Update tracker.
        # print("Time taken for everything before updating metadata", time.time() - start)
        start = time.time()
        tracker.update_metadata(frame_idx, depth_map_path,
                                seq_info, max_height,
                                only_filtering=only_filtering,
                                temporal_noise=temporal_noise,
                                ah_velocity=ah_velocity,
                                velocity_weighting=velocity_weighting,
                                tn=tn,
                                motion_aware=motion_aware)
        # print("Time taken to update metadata", time.time() - start)
        start = time.time()
        tracker.predict()
        # print("Time taken to predict", time.time() - start)
        start = time.time()
        # print(len(detections))
        tracker.update(detections,
                       default_matching=default_matching,
                       freespace_filtering=freespace_filtering,
                       occluded_factor=occluded_factor,
                       filtering_factor=filtering_factor,
                       only_extrapolate=only_extrapolate,
                       extrapolated_iou_match=extrapolated_iou_match,
                       appearance_match=appearance_match,
                       bugfix=bugfix)
        # print("Time taken to update", time.time() - start)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if track.is_tentative() or track.is_deleted():
                continue
            if not only_filtering and (track.is_confirmed() and track.time_since_update > 1):
                continue
            if output_uncertainty:
                bbox = track.to_tlwhz_cov()
            else:
                bbox = track.to_tlwhz()
            if track.is_confirmed():
                if output_uncertainty:
                    results.append([frame_idx, track.track_id, bbox[0],
                                bbox[1], bbox[2], bbox[3], bbox[4], 0,
                                bbox[5], bbox[6], bbox[7], bbox[8]])
                    # print([frame_idx, track.track_id, bbox[0],
                    #             bbox[1], bbox[2], bbox[3], bbox[4], 0,
                    #             bbox[5], bbox[6], bbox[7], bbox[8]])
                else:
                    results.append([frame_idx, track.track_id, bbox[0],
                                bbox[1], bbox[2], bbox[3], bbox[4], 0])
            else:
                # print("Tracker was in the occluded state")
                if output_uncertainty:
                    results.append([frame_idx, track.track_id, bbox[0],
                                bbox[1], bbox[2], bbox[3], bbox[4], 1,
                                bbox[5], bbox[6], bbox[7], bbox[8]])
                else:
                    results.append([frame_idx, track.track_id, bbox[0],
                                bbox[1], bbox[2], bbox[3], bbox[4], 1])
        # print("Time taken to run everything on a frame", time.time() - startall)

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        if output_uncertainty:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1,%d,%.4f,%.4f,%.4f,%.4f' % (
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[-5], row[-4],
                row[-3], row[-2], row[-1]),file=f)
        else:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1,%d' % (
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[-1]),file=f)


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    parser.add_argument(
        "--temporal_noise", help="Show intermediate tracking results",
        default=True, type=bool_string)
    parser.add_argument(
        "--only_filtering", help="Show intermediate tracking results",
        default=False, type=bool_string)
    parser.add_argument(
        "--default_matching", help="Show intermediate tracking results",
        default=False, type=bool_string)
    parser.add_argument(
        "--freespace_filtering", help="Show intermediate tracking results",
        default=True, type=bool_string)
    parser.add_argument(
        "--ah_velocity", help="Show intermediate tracking results",
        default=False, type=bool_string)
    parser.add_argument(
        "--velocity_weighting", help="Show intermediate tracking results",
        default=True, type=bool_string)
    parser.add_argument(
        "--motion-aware", help="Show intermediate tracking results",
        default=False, type=bool_string)
    parser.add_argument(
        "--output-uncertainty", help="Show intermediate tracking results",
        default=False, type=bool_string)
    parser.add_argument(
        "--only-extrapolate", help="Show intermediate tracking results",
        default=False, type=bool_string)
    parser.add_argument(
        "--extrapolated-iou-match", help="Show intermediate tracking results",
        default=False, type=bool_string)
    parser.add_argument(
        "--appearance-match", help="Show intermediate tracking results",
        default=True, type=bool_string)
    parser.add_argument(
        "--bugfix", help="Show intermediate tracking results",
        default=False, type=bool_string)
    parser.add_argument(
        "--tune_temporal_noise", help="Show intermediate tracking results",
        default=False, type=bool_string)
    parser.add_argument(
        "--obs_constant", help="Show intermediate tracking results",
        default=500, type=int)
    parser.add_argument(
        "--obs_factor", help="Show intermediate tracking results",
        default=1, type=int)
    parser.add_argument(
        "--proc_constant", help="Show intermediate tracking results",
        default=500, type=int)
    parser.add_argument(
        "--proc_factor", help="Show intermediate tracking results",
        default=1, type=int)
    parser.add_argument(
        "--occluded_factor", help="Show intermediate tracking results",
        default=1.0, type=float)
    parser.add_argument(
        "--filtering_factor", help="Show intermediate tracking results",
        default=1.0, type=float)
    parser.add_argument(
        "--max-age", help="Maximum frames to keep a track alive even when it "
        "hasn't found a matching groundtruth detection.", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sequences = os.listdir(args.sequence_dir)
    print(sequences)
    if args.tune_temporal_noise:
        tn = {'oc': args.obs_constant,
              'of': args.obs_factor,
              'pc': args.proc_constant,
              'pf': args.proc_factor}
    else:
        tn = -1
    for seq in sequences:
        print(seq)
        detection_file = "/data/tkhurana/deep_sort/resources/detections/MOT17_train/" + seq + ".npy"
        output_file = "/data/tkhurana/tk/deep_sort/results/" + args.output_file + "/{}.txt".format(seq)
        if os.path.exists(output_file):
            continue
        sequence_dir = args.sequence_dir + "/" + seq
        depth_map_path = args.sequence_dir + "/" + seq.replace("DPM", "SDP").replace("SDP", "FRCNN")
        run(
            sequence_dir, detection_file, output_file, args.min_confidence,
            args.nms_max_overlap, args.min_detection_height, args.max_cosine_distance,
            args.nn_budget, args.display, depth_map_path, args.max_age, args.only_filtering,
            args.temporal_noise, args.default_matching, args.freespace_filtering,
            args.ah_velocity, args.velocity_weighting, tn, args.occluded_factor,
            args.filtering_factor, args.motion_aware, args.output_uncertainty,
            args.only_extrapolate, args.extrapolated_iou_match, args.appearance_match,
            args.bugfix)


# bash run_forecast_filtering.sh 30 temp2 0.3 1 0.5 False True False True True False True 500 1 500 1 1.0 1.0
# bash run_forecast_filtering.sh 30 temp 0.3 1 0.5 False False True True True False True 500 1 500 1 1.0 1.0
# bash run_forecast_filtering.sh 30 temp3 0.3 1 0.5 False False True False True False True 500 1 500 1 1.0 1.0
