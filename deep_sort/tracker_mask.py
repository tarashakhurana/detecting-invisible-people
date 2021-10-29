# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from skimage.filters import threshold_otsu
import os
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from pycocotools import mask as maskUtils
import cv2
from skimage.transform import resize
from PIL import Image

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.structures import Boxes, Instances
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from pycocotools import mask as maskUtils

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args['config_file'])
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args['confidence_threshold']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args['confidence_threshold']
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args['confidence_threshold']
    cfg.MODEL.WEIGHTS = 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl'
    cfg.freeze()
    return cfg

def get_parser():
    parser = {'config_file': '/home/tkhurana/CVPR/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml', 'confidence_threshold': 0.5}
    return parser

def sort_to_detectron2(detections):
    boxes = Boxes(torch.from_numpy(np.asarray(detections)))
    return boxes


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.frame_idx = -1
        self.depth_map_path = ''
        self.sequence_info = {}
        self.max_height = -1
        self.image = []
        self.tn = -1
        self.past_frame = []
        self.current_frame = []
        self.warp_matrix = -1

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

        self.vicinity_x = 25
        self.vicinity_y = 0

    def get_masks(self):
        bboxes = []

        for track in self.tracks:
            x, y, w, h = track.to_tlwh()
            bboxes.append([x, y, x+w, y+h])

        impath = os.path.join(
                    self.depth_map_path,
                    'img1',
                    '{:06d}.jpg'.format(self.frame_idx))
        if len(bboxes) != 0:
            self.masks = self.get_mask_for_bbox(bboxes, impath)
        else:
            self.masks = []

    def get_mask_for_bbox(self, bboxes, path):
        width, height = Image.open(path).size
        j = 0
        mask_array = []
        while j < len(bboxes):
            bbox_mask = np.zeros((height, width), dtype='uint8')
            x1, y1, x2, y2 = bboxes[j]
            bbox_mask[int(y1):int(y2), int(x1):int(x2)] = 1
            mask_array.append(bbox_mask)
            j += 1

        return mask_array

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        # print("Len of tracks:", len(self.tracks))
        for track in self.tracks:
            track.predict(self.kf, self.max_height, tn=self.tn,
                          warp_matrix=self.warp_matrix)

    def update(self, detections, occluded_factor=1.0,
               filtering_factor=1.0):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections, newly_occluded_tracks, previously_occluded_tracks = \
            self._match(detections, occluded_factor=occluded_factor, filtering_factor=filtering_factor)

        # use this with only_filtering True and default_matching False to get just deepsort+
        # extrapolate+depth; the filtered out boxes should be joined back to unmatched_tracks if
        # this flag is true.
        if only_extrapolate:
            unmatched_tracks = unmatched_tracks + previously_occluded_tracks
            previously_occluded_tracks = []

        # for all the matched detection and track pairs, we are going to (conditionally) call
        # these confirmed tracks and do the needful (as you can find in the update function in
        # track.py in this folder).
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx],
                self.image, self.sequence_info,
                temporal_noise=self.temporal_noise,
                tn=self.tn)

        # for all the newly_occluded_tracks, we are going to call these occluded if they
        # were previously a confirmed track. if these tracks are still occluded and it has
        # been > max_age then we are going to delete these tracks.
        for track_idx in newly_occluded_tracks:
            self.tracks[track_idx].mark_occluded()

        # these are the tracks that got filtered due to freespace filtering so take a hard
        # decision of deleting these.
        for track_idx in previously_occluded_tracks:
            self.tracks[track_idx].mark_deleted()

        # for the tracks that were in confirmed state but which were left unmatched, delete
        # them if it has been > max_age.
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # for all unmatched detections in the current frame, start a new track.
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx],
            temporal_noise=self.temporal_noise, tn=self.tn)

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed() or t.is_occluded()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed() and not track.is_occluded():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections, default_matching=False,
               freespace_filtering=True, occluded_factor=1.0,
               filtering_factor=1.0, extrapolated_iou_match=False,
               appearance_match=True, bugfix=False):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices, temporal_noise=self.temporal_noise,
                tn=self.tn)

            return cost_matrix

        self.get_masks()

        # Split track set into confirmed, occluded and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        occluded_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_occluded()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed() and not t.is_occluded()]

        # find all occluded tracks from the set of confirmed tracks and collectively
        # call them newly_occluded_tracks. the set of tracks that were not occluded will
        # still be in confirmed_tracks.
        if not self.only_filtering:
            newly_occluded_tracks, confirmed_tracks = self.reason_for_occlusions_mask(
                                                        self.tracks,
                                                        confirmed_tracks,
                                                        occluded_factor)
            newly_occluded_tracks = newly_occluded_tracks + occluded_tracks

        # if using default matching, merge all kinds of tracks together into confirmed_tracks
        # and match these together based on appearance. later we will segregate them again
        if not self.only_filtering and default_matching and appearance_match:
            confirmed_tracks = confirmed_tracks + newly_occluded_tracks
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    gated_metric, self.metric.matching_threshold, 0, self.max_age,
                    self.tracks, detections, confirmed_tracks)
        elif not self.only_filtering and default_matching and not appearance_match:
            confirmed_tracks = confirmed_tracks + newly_occluded_tracks
            matches_a = []
            unmatched_tracks_a = confirmed_tracks
            unmatched_detections = [idx for idx, det in enumerate(detections)]

        # similar, except we dont match the confirmed and occluded tracks together now
        if not default_matching and appearance_match:
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    gated_metric, self.metric.matching_threshold, 0, self.max_age,
                    self.tracks, detections, confirmed_tracks)
        elif not default_matching and not appearance_match:
            matches_a = []
            unmatched_tracks_a = confirmed_tracks
            unmatched_detections = [idx for idx, det in enumerate(detections)]

        # similar idea, above was for matching confirmed tracks, now we are matching the
        # occluded tracks. in this case, the occluded tracks that actually got matched to
        # a detection, we should call it a confirmed track now and the ones that didnt match
        # should still be in the occluded state.
        if not self.only_filtering and not default_matching and appearance_match:
            # print("matching c!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            matches_c, newly_occluded_tracks, unmatched_detections = \
                linear_assignment.matching_cascade(
                    gated_metric, self.metric.matching_threshold, 0, self.max_age, # 0.15
                    self.tracks, detections, newly_occluded_tracks, unmatched_detections)
        elif not self.only_filtering and not default_matching and not appearance_match:
            matches_c = []

        # for track_idx, detection_idx in matches_a:
        #     if self.tracks[track_idx].track_id == 4:
        #         print("track was matched in a!!!!!!!!!!!!")

        # for track_idx, detection_idx in matches_c:
        #     if self.tracks[track_idx].track_id == 4:
        #         print("track was matched in c!!!!!!!!!!!!")

        # this is an original step in deepsort
        # Associate remaining tracks together with unconfirmed tracks using IOU.

        # extrapolated iou match debug
        # temp = [k for k in unmatched_tracks_a if
        #         self.tracks[k].time_since_update != 1 \
        #             and self.tracks[k].state == 4]
        # print("debug print", temp)

        if extrapolated_iou_match:
            # print("Extrapolated iou match was true")
            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a]
            unmatched_tracks_a = []
        else:
            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update == 1]
            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        # extrapolated iou match debug
        # print("iou matches", matches_b)

        # very trivial, just takes care of whether we have three sets of matches till
        # now or only two
        if not self.only_filtering and not default_matching:
            matches = matches_a + matches_b + matches_c # + matches_d
        else:
            matches = matches_a + matches_b # + matches_c # + matches_d

        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        # this step segregates the occluded tracks from the unmatched confirmed tracks
        # if you used default matching above, (because we merged both into one for default
        # matching)
        if default_matching:
            newly_occluded_tracks = [i for i in newly_occluded_tracks if i in unmatched_tracks]
            unmatched_tracks = [i for i in unmatched_tracks if i not in newly_occluded_tracks]

        # if we weren't using occluded state, then we havent formed any variable called
        # newly_occluded_tracks yet, so just call unmatched_tracks as this for the next step
        if self.only_filtering and not default_matching:
            newly_occluded_tracks = unmatched_tracks

        # either do freespace filtering or if we werent supposed to filter, then there is no
        # notion of previously_occluded_tracks (these are the set of tracks that were filtered
        # so they are going to be deleted if stored in this variable) and all newly_occluded_tracks
        # are still maintained in the occluded state
        if (freespace_filtering or self.only_filtering) and not default_matching:
            previously_occluded_tracks, occluded_tracks_ = self.reason_for_reappearances_mask(
                                                        self.tracks,
                                                        newly_occluded_tracks,
                                                        filtering_factor)
        elif (freespace_filtering or self.only_filtering) and default_matching and bugfix:
            # print("Executing bugfix")
            pv1, occluded_tracks_ = self.reason_for_reappearances_mask(
                                                        self.tracks,
                                                        newly_occluded_tracks,
                                                        filtering_factor)
            pv2, unmatched_tracks = self.reason_for_reappearances_mask(
                                                        self.tracks,
                                                        unmatched_tracks,
                                                        filtering_factor)
            previously_occluded_tracks = pv1 + pv2
        elif (freespace_filtering or self.only_filtering) and default_matching and not bugfix:
            # print("Not executing bugfix")
            previously_occluded_tracks, occluded_tracks_ = self.reason_for_reappearances_mask(
                                                        self.tracks,
                                                        newly_occluded_tracks,
                                                        filtering_factor)
        else:
            previously_occluded_tracks = []
            occluded_tracks_ = newly_occluded_tracks

        # if we were only filtering, then there was no notion of occluded_tracks_ and these are
        # actually the tracks that did not get filtered and so really, are still unmatched
        if self.only_filtering and not default_matching:
            unmatched_tracks = occluded_tracks_
            occluded_tracks_ = []

        # two caveats: one, some variables or if statements might be redundant, pls excuse my
        # coding, two, because of this reason, always have to take care that if only_filtering is
        # set to true then default_matching should be set to false for the code to execute properly
        # print("matches, unmatched tracks, unmatched detections, occluded_tracks_, previously_occluded_tracks",
        #       len(matches), len(unmatched_tracks), len(unmatched_detections),
        #       len(occluded_tracks_), len(previously_occluded_tracks))
        return matches, unmatched_tracks, unmatched_detections, occluded_tracks_, previously_occluded_tracks

    # DO NOT TRUST THIS CODE
    def _match_swap(self, detections, default_matching=False,
               freespace_filtering=True, occluded_factor=1.0,
               filtering_factor=1.0, extrapolated_iou_match=False,
               appearance_match=True, bugfix=False):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            print("detection indices", detection_indices)
            print("track indices", track_indices)
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices, temporal_noise=self.temporal_noise,
                tn=self.tn)

            return cost_matrix

        # Split track set into confirmed, occluded and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        occluded_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_occluded()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed() and not t.is_occluded()]

        # find all occluded tracks from the set of confirmed tracks and collectively
        # call them newly_occluded_tracks. the set of tracks that were not occluded will
        # still be in confirmed_tracks.
        if not self.only_filtering:
            newly_occluded_tracks, confirmed_tracks = self.reason_for_occlusions(
                                                        self.tracks,
                                                        confirmed_tracks,
                                                        occluded_factor)
            newly_occluded_tracks = newly_occluded_tracks + occluded_tracks

        # if using default matching, merge all kinds of tracks together into confirmed_tracks
        # and match these together based on appearance. later we will segregate them again
        if not self.only_filtering and default_matching: # and appearance_match:
            confirmed_tracks = confirmed_tracks + newly_occluded_tracks + unconfirmed_tracks
            matches_b, unmatched_tracks_b, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                    detections, confirmed_tracks)

        # similar, except we dont match the confirmed and occluded tracks together now
        if not default_matching: # and appearance_match:
            matches_b, unmatched_tracks_b, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                    detections, confirmed_tracks)

        # similar idea, above was for matching confirmed tracks, now we are matching the
        # occluded tracks. in this case, the occluded tracks that actually got matched to
        # a detection, we should call it a confirmed track now and the ones that didnt match
        # should still be in the occluded state.
        if not self.only_filtering and not default_matching: # and appearance_match:
            matches_c, newly_occluded_tracks, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                    detections, newly_occluded_tracks, unmatched_detections)

        iou_track_candidates = unmatched_tracks_b
        unmatched_tracks_b = []

        # print(len(iou_track_candidates), len(unmatched_detections))

        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, 0, self.max_age,
                self.tracks, detections, iou_track_candidates, unmatched_detections)

        # very trivial, just takes care of whether we have three sets of matches till
        # now or only two
        if not self.only_filtering and not default_matching:
            matches = matches_a + matches_b + matches_c # + matches_d
        else:
            matches = matches_a + matches_b # + matches_c # + matches_d

        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        # this step segregates the occluded tracks from the unmatched confirmed tracks
        # if you used default matching above, (because we merged both into one for default
        # matching)
        if default_matching:
            newly_occluded_tracks = [i for i in newly_occluded_tracks if i in unmatched_tracks]
            unmatched_tracks = [i for i in unmatched_tracks if i not in newly_occluded_tracks]

        # if we weren't using occluded state, then we havent formed any variable called
        # newly_occluded_tracks yet, so just call unmatched_tracks as this for the next step
        if self.only_filtering and not default_matching:
            newly_occluded_tracks = unmatched_tracks

        # either do freespace filtering or if we werent supposed to filter, then there is no
        # notion of previously_occluded_tracks (these are the set of tracks that were filtered
        # so they are going to be deleted if stored in this variable) and all newly_occluded_tracks
        # are still maintained in the occluded state
        if (freespace_filtering or self.only_filtering) and not default_matching:
            previously_occluded_tracks, occluded_tracks_ = self.reason_for_reappearances(
                                                        self.tracks,
                                                        newly_occluded_tracks,
                                                        filtering_factor)
        elif (freespace_filtering or self.only_filtering) and default_matching and bugfix:
            # print("Executing bugfix")
            pv1, occluded_tracks_ = self.reason_for_reappearances(
                                                        self.tracks,
                                                        newly_occluded_tracks,
                                                        filtering_factor)
            pv2, unmatched_tracks = self.reason_for_reappearances(
                                                        self.tracks,
                                                        unmatched_tracks,
                                                        filtering_factor)
            previously_occluded_tracks = pv1 + pv2
        elif (freespace_filtering or self.only_filtering) and default_matching and not bugfix:
            # print("Not executing bugfix")
            previously_occluded_tracks, occluded_tracks_ = self.reason_for_reappearances(
                                                        self.tracks,
                                                        newly_occluded_tracks,
                                                        filtering_factor)
        else:
            previously_occluded_tracks = []
            occluded_tracks_ = newly_occluded_tracks

        # if we were only filtering, then there was no notion of occluded_tracks_ and these are
        # actually the tracks that did not get filtered and so really, are still unmatched
        if self.only_filtering and not default_matching:
            unmatched_tracks = occluded_tracks_
            occluded_tracks_ = []

        # two caveats: one, some variables or if statements might be redundant, pls excuse my
        # coding, two, because of this reason, always have to take care that if only_filtering is
        # set to true then default_matching should be set to false for the code to execute properly
        # print("matches, unmatched tracks, unmatched detections, occluded_tracks_, previously_occluded_tracks",
        #       len(matches), len(unmatched_tracks), len(unmatched_detections),
        #       len(occluded_tracks_), len(previously_occluded_tracks))
        return matches, unmatched_tracks, unmatched_detections, occluded_tracks_, previously_occluded_tracks


    def _initiate_track(self, detection, temporal_noise=True, tn=-1):
        mean_depth = self.compute_mean_depth_from_mask(self.image, detection, self.sequence_info)
        # print(mean_depth)
        det = list(detection.to_xyah())
        det = det + [mean_depth]
        mean, covariance = self.kf.initiate(det, temporal_noise, tn)
        self.tracks.append(Track(
            mean, covariance, self._next_id,
            self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1

    def compute_mean_depth(self, depth_map, detection, seq_info):
        scale_x = seq_info["image_size"][1] / float(depth_map.shape[1])
        scale_y = seq_info["image_size"][0] / float(depth_map.shape[0])
        box = detection.tlwh.copy()
        box[2:] += box[:2]

        box = [box[0]/scale_x,
               box[1]/scale_y,
               box[2]/scale_x,
               box[3]/scale_y]
        box = [int(x) for x in box]
        box = [max(0, box[0]), max(0, box[1]),
               max(0, min(depth_map.shape[1], box[2])),
               max(0, min(depth_map.shape[0], box[3]))]

        if 0 in box[2:] \
            or box[0] >= depth_map.shape[1] \
            or box[1] >= depth_map.shape[0] \
            or box[0] == box[2] \
            or box[1] == box[2]:
            return -1

        box = depth_map[box[1]:box[3], box[0]:box[2]].copy()
        return np.mean(box)

    def compute_mean_depth_from_mask(self, depth_map, detection, seq_info, mask=None):
        width = depth_map.shape[1]
        height = depth_map.shape[0]

        # print(detection.mask['counts'], detection.mask['size'])
        if detection is not None:
            m = detection.mask.copy()
        elif mask is not None:
            m = mask
        else:
            print("One of detection or mask has to be non-None")
            exit(0)

        m = resize(m, (height, width), order=1)

        inter_mask = np.zeros((height, width), dtype=float)
        inter_mask = np.where(m > 10e-6, depth_map, 0)

        if 0 in np.nonzero(inter_mask)[0].shape:
            return -1
        return np.mean(inter_mask[np.nonzero(inter_mask)])


    def align(self, im1_gray, im2_gray):
        # maximal number of iterations (original 50)
        number_of_iterations = 50 # 100
        # Threshold increment between two iterations (original 0.001)
        termination_eps = 0.001 # 0.00001
        # Which warp mode to use (cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE, ...)
        warp_mode = cv2.MOTION_EUCLIDEAN

        # im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        # im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    number_of_iterations,
                    termination_eps)
        try:
            cc, warp_matrix = cv2.findTransformECC(im1_gray,
                    im2_gray, warp_matrix,
                    warp_mode, criteria,
                    inputMask=None, gaussFiltSize=1)
        except TypeError:
            cc, warp_matrix = cv2.findTransformECC(im1_gray,
                    im2_gray, warp_matrix,
                    warp_mode, criteria)


        # if self.do_reid:
        #     for t in self.inactive_tracks:
        #         t.pos = warp_pos(t.pos, warp_matrix)

        # if self.motion_model_cfg['enabled']:
        #     for t in self.tracks:
        #         for i in range(len(t.last_pos)):
        #             t.last_pos[i] = warp_pos(t.last_pos[i], warp_matrix)

        return warp_matrix

    def update_metadata(self, idx, path, seq_info, max_height,
                        only_filtering=False, temporal_noise=True,
                        ah_velocity=False, velocity_weighting=True,
                        tn=-1, motion_aware=False):
        self.frame_idx = idx
        self.depth_map_path = path
        self.sequence_info = seq_info
        self.max_height = max_height
        self.image = np.load(
            os.path.join(
                self.depth_map_path,
                'img1Depth',
                '{:06d}.npy'.format(self.frame_idx)))
        if self.frame_idx != 1:
            self.past_frame = cv2.imread(
                os.path.join(
                    self.depth_map_path,
                    'img1',
                    '{:06d}.jpg'.format(self.frame_idx - 1)),
                0
            )
        self.current_frame = cv2.imread(
                os.path.join(
                    self.depth_map_path,
                    'img1',
                    '{:06d}.jpg'.format(self.frame_idx)),
                0
            )
        self.only_filtering = only_filtering
        self.temporal_noise = temporal_noise
        self.ah_velocity = ah_velocity
        self.velocity_weighting = velocity_weighting
        self.tn = tn
        if self.frame_idx != 1 and motion_aware:
            # print("aligning ...")
            warp_path = os.path.join(
                    self.depth_map_path,
                    'warpmatrix',
                    '{:06d}.npy'.format(self.frame_idx))
            if os.path.exists(warp_path):
                self.warp_matrix = np.load(warp_path)
            else:
                os.makedirs(os.path.dirname(warp_path), exist_ok=True)
                self.warp_matrix = self.align(self.past_frame,
                                      self.current_frame)
                np.save(warp_path, self.warp_matrix)
        self.motion_aware = motion_aware

    def reason_for_occlusions(self, tracks, track_indices, occluded_factor=1.0):
        if self.frame_idx == -1:
            return [], track_indices

        newly_occluded_tracks, unmatched_tracks = [], []
        image = self.image.copy() # np.load(os.path.join(self.depth_map_path, 'img1Depth',
                                    #     '{:06d}.npy'.format(self.frame_idx)))
        scale_x = self.sequence_info["image_size"][1] / float(image.shape[1])
        scale_y = self.sequence_info["image_size"][0] / float(image.shape[0])

        for idx in track_indices:
            track = self.tracks[idx]
            box = track.to_tlbr()
            _, _, _, _, predicted_depth = track.to_tlwhz()
            box = [box[0]/scale_x, box[1]/scale_y, box[2]/scale_x, box[3]/scale_y]
            box = [int(x) for x in box]
            box = [max(0, box[0]), max(0, box[1]),
                   max(min(image.shape[1], box[2]), 0),
                   max(min(image.shape[0], box[3]), 0)]

            if 0 in box[2:] or box[0] >= image.shape[1] or box[1] >= image.shape[0] or box[0] == box[2] or box[1] == box[2]:
                unmatched_tracks.append(idx)
                continue

            box = image[box[1]:box[3], box[0]:box[2]].copy()

            if len(np.unique(box)) == 1:
                unmatched_tracks.append(idx)
                continue
            if 0 in box.shape:
                unmatched_tracks.append(idx)
                continue

            box_mean = np.mean(box[np.nonzero(box)])

            # if track.track_id == 4:
            #     print("two depths are", predicted_depth, box_mean)

            if predicted_depth * occluded_factor < box_mean:
                newly_occluded_tracks.append(idx)
            else:
                unmatched_tracks.append(idx)

        return newly_occluded_tracks, unmatched_tracks


    def reason_for_reappearances(self, tracks, track_indices, filtering_factor=1.0):
        if self.frame_idx == -1:
            return [], track_indices

        previously_occluded_tracks, occluded_tracks = [], []
        image = self.image.copy() # np.load(os.path.join(self.depth_map_path, 'img1Depth',
                                    #     '{:06d}.npy'.format(self.frame_idx)))
        scale_x = self.sequence_info["image_size"][1] / float(image.shape[1])
        scale_y = self.sequence_info["image_size"][0] / float(image.shape[0])

        for idx in track_indices:
            track = self.tracks[idx]
            box = track.to_tlbr()
            _, _, _, _, predicted_depth = track.to_tlwhz()
            box = [box[0]/scale_x, box[1]/scale_y, box[2]/scale_x, box[3]/scale_y]
            box = [int(x) for x in box]
            box = [max(0, box[0]), max(0, box[1]),
                   max(min(image.shape[1], box[2]), 0),
                   max(min(image.shape[0], box[3]), 0)]

            if 0 in box[2:] or box[0] >= image.shape[1] or box[1] >= image.shape[0] or box[0] == box[2] or box[1] == box[2]:
                occluded_tracks.append(idx)
                continue

            box = image[box[1]:box[3], box[0]:box[2]].copy()

            if len(np.unique(box)) == 1:
                occluded_tracks.append(idx)
                continue
            if 0 in box.shape:
                occluded_tracks.append(idx)
                continue

            box_mean = np.mean(box[np.nonzero(box)])

            # if track.track_id == 4:
            #     print("in filtering, two depths are", predicted_depth, box_mean)

            if predicted_depth > box_mean * filtering_factor:
                previously_occluded_tracks.append(idx)
            else:
                occluded_tracks.append(idx)

        return previously_occluded_tracks, occluded_tracks

    def reason_for_occlusions_mask(self, tracks, track_indices, occluded_factor=1.0):
        if self.frame_idx == -1:
            return [], track_indices

        newly_occluded_tracks, unmatched_tracks = [], []
        image = self.image.copy() # np.load(os.path.join(self.depth_map_path, 'img1Depth',
                                    #     '{:06d}.npy'.format(self.frame_idx)))

        for idx in track_indices:
            track = self.tracks[idx]
            _, _, _, _, predicted_depth = track.to_tlwhz()

            box_mean = self.compute_mean_depth_from_mask(
                    image, None, self.sequence_info, self.masks[idx])

            if predicted_depth * occluded_factor < box_mean:
                newly_occluded_tracks.append(idx)
            else:
                unmatched_tracks.append(idx)

        return newly_occluded_tracks, unmatched_tracks


    def reason_for_reappearances_mask(self, tracks, track_indices, filtering_factor=1.0):
        if self.frame_idx == -1:
            return [], track_indices

        previously_occluded_tracks, occluded_tracks = [], []
        image = self.image.copy() # np.load(os.path.join(self.depth_map_path, 'img1Depth',
                                    #     '{:06d}.npy'.format(self.frame_idx)))

        for idx in track_indices:
            track = self.tracks[idx]
            _, _, _, _, predicted_depth = track.to_tlwhz()

            box_mean = self.compute_mean_depth_from_mask(
                    image, None, self.sequence_info, self.masks[idx])

            if predicted_depth > box_mean * filtering_factor:
                previously_occluded_tracks.append(idx)
            else:
                occluded_tracks.append(idx)

        return previously_occluded_tracks, occluded_tracks


############################################################################################################
############################################################################################################
############################################################################################################

    def reason_for_occlusions_old(self, tracks, track_indices, noise=0.98):
        # print(len(self.tracks))
        if self.frame_idx == -1:
            return [], track_indices

        # Use depth to find potentially occluded tracks
        newly_occluded_tracks, unmatched_tracks = [], []
        image = self.image.copy() # np.load(os.path.join(self.depth_map_path, 'img1Depth',
                                    #     '{:06d}.npy'.format(self.frame_idx)))

        scale_x = self.sequence_info["image_size"][1] / float(image.shape[1])
        scale_y = self.sequence_info["image_size"][0] / float(image.shape[0])

        for idx in track_indices:
            track = self.tracks[idx]
            # predicted, _ = track.predict(self.kf, self.max_height, update_age=False)

            # ret = predicted[:4]
            # ret[2] *= ret[3]
            # ret[:2] -= ret[2:] / 2
            # ret[2:] = ret[:2] + ret[2:]
            # print("Doing track", track.track_id)
            # if track.track_id == 14:
            #     print("Doing this track", idx)
            img = image.copy() * 255



            # crop out the original and extended boxes from the depth map
            box = track.to_tlbr()
            # print("box1", box)
            box = [box[0]/scale_x, box[1]/scale_y, box[2]/scale_x, box[3]/scale_y]
            # print("box2", box, scale_x, scale_y)
            box = [int(x) for x in box]
            box_vicinity = [box[0] - self.vicinity_x, box[1] - self.vicinity_y,
                            box[2] + self.vicinity_x, box[3] + self.vicinity_y]

            box = [max(0, box[0]), max(0, box[1]),
                   max(min(image.shape[1], box[2]), 0),
                   max(min(image.shape[0], box[3]), 0)]
            box_vicinity = [max(0, box_vicinity[0]), max(0, box_vicinity[1]),
                            max(0, min(image.shape[1], box_vicinity[2])),
                            max(0, min(image.shape[0], box_vicinity[3]))]

            boxx = box
            boxx_vicinity = box_vicinity

            # print(box, box_vicinity, image.shape[1], image.shape[0])

            if 0 in box[2:] or 0 in box_vicinity[2:] or box[0] >= image.shape[1] or box_vicinity[0] >= image.shape[1] or box[1] >= image.shape[0] or box_vicinity[1] >= image.shape[0] or box[0] == box[2] or box[1] == box[2]:
                # print("Skipping ...", track.track_id)
                # if track.track_id == 30:
                #     print(box, box_vicinity)
                #     print("Skipping from 1")
                unmatched_tracks.append(idx)
                continue


            box = image[box[1]:box[3], box[0]:box[2]].copy()
            box_vicinity = image[box_vicinity[1]:box_vicinity[3],
                                 box_vicinity[0]:box_vicinity[2]].copy()

            if len(np.unique(box)) == 1 or len(np.unique(box_vicinity)) == 1:
                unmatched_tracks.append(idx)
                # if track.track_id == 30:
                #     print("Skipping from 1")
                continue

            if 0 in box.shape or 0 in box_vicinity.shape:
                unmatched_tracks.append(idx)
                continue

            # img = cv2.rectangle(img, (boxx[0], boxx[1]), (boxx[2], boxx[3]), (0, 0, 0), 1)
            # cv2.rectangle(img, (boxx_vicinity[0], boxx_vicinity[1]), (boxx_vicinity[2], boxx_vicinity[3]), (0, 0, 0), 1)

            # calculate the Otsu's threshold and get all important pixels above this threshold from
            # both the original and the extended boxes so we can reason if there is an object closer
            # than the current object represented by these important pixels
            # if not os.path.exists('/data/tkhurana/tk/deep_sort/verificatio/{}/'.format(track.track_id)):
            #     os.makedirs('/data/tkhurana/tk/deep_sort/verificatio/{}/'.format(track.track_id))
            # cv2.imwrite('/data/tkhurana/tk/deep_sort/verificatio/{}/{}_boxes.jpg'.format(track.track_id, self.frame_idx), img)
            # cv2.imwrite('/data/tkhurana/tk/deep_sort/verification/{}/{}_box_vicinity.jpg'.format(track.track_id, self.frame_idx), box_vicinity * 255)

            thresh = threshold_otsu(box)
            box_pixels = box * (box > thresh)
            # cv2.imwrite('/data/tkhurana/tk/deep_sort/verificatio/{}/{}_box_pixels.jpg'.format(track.track_id, self.frame_idx), box_pixels * 255)
            box_vicinity_pixels = box_vicinity * (box_vicinity > thresh)
            # cv2.imwrite('/data/tkhurana/tk/deep_sort/verificatio/{}/{}_box_vicinity_pixels.jpg'.format(track.track_id, self.frame_idx), box_vicinity_pixels * 255)
            box_mean = np.mean(box_pixels[np.nonzero(box_pixels)])
            box_vicinity_mean = np.mean(box_vicinity_pixels[np.nonzero(box_vicinity_pixels)])

            # if track.track_id == 30:
            #     print(box_vicinity_mean, box_mean, box_mean * noise)

            if box_vicinity_mean > box_mean * noise:
                # if track.track_id == 8:
                #     print("was here", idx)
                newly_occluded_tracks.append(idx)
            else:
                unmatched_tracks.append(idx)


        return newly_occluded_tracks, unmatched_tracks




    def reason_for_reappearances_old(self, tracks, track_indices, noise=0.75):
        # print(len(self.tracks))
        if self.frame_idx == -1:
            return [], track_indices

        # Use depth to find potentially occluded tracks
        previously_occluded_tracks, unmatched_tracks = [], []
        image = self.image.copy()

        scale_x = self.sequence_info["image_size"][1] / float(image.shape[1])
        scale_y = self.sequence_info["image_size"][0] / float(image.shape[0])

        for idx in track_indices:
            track = self.tracks[idx]
            img = image.copy() * 255

            # crop out the original and extended boxes from the depth map
            box = track.to_tlbr()
            box = [box[0]/scale_x, box[1]/scale_y, box[2]/scale_x, box[3]/scale_y]
            box = [int(x) for x in box]
            # box_vicinity = [box[0] - self.vicinity_x, box[1] - self.vicinity_y,
            #                 box[2] + self.vicinity_x, box[3] + self.vicinity_y]

            box = [max(0, box[0]), max(0, box[1]),
                   max(min(image.shape[1], box[2]), 0),
                   max(min(image.shape[0], box[3]), 0)]
            # box_vicinity = [max(0, box_vicinity[0]), max(0, box_vicinity[1]),
            #                 max(0, min(image.shape[1], box_vicinity[2])),
            #                 max(0, min(image.shape[0], box_vicinity[3]))]

            boxx = box
            # boxx_vicinity = box_vicinity

            # print(box, box_vicinity, image.shape[1], image.shape[0])

            if 0 in box[2:] or box[0] >= image.shape[1] or box[1] >= image.shape[0] or box[0] == box[2] or box[1] == box[2]:
                unmatched_tracks.append(idx)
                continue

            box = image[box[1]:box[3], box[0]:box[2]].copy()
            # box_vicinity = image[box_vicinity[1]:box_vicinity[3],
            #                      box_vicinity[0]:box_vicinity[2]].copy()

            if len(np.unique(box)) == 1:
                unmatched_tracks.append(idx)
                continue

            if 0 in box.shape:
                unmatched_tracks.append(idx)
                continue

            # img = cv2.rectangle(img, (boxx[0], boxx[1]), (boxx[2], boxx[3]), (0, 0, 0), 1)
            # cv2.rectangle(img, (boxx_vicinity[0], boxx_vicinity[1]), (boxx_vicinity[2], boxx_vicinity[3]), (0, 0, 0), 1)

            # if not os.path.exists('/data/tkhurana/tk/deep_sort/verificatio/{}/'.format(track.track_id)):
            #     os.makedirs('/data/tkhurana/tk/deep_sort/verificatio/{}/'.format(track.track_id))
            # cv2.imwrite('/data/tkhurana/tk/deep_sort/verificatio/{}/{}_boxes.jpg'.format(track.track_id, self.frame_idx), img)
            # cv2.imwrite('/data/tkhurana/tk/deep_sort/verification/{}/{}_box_vicinity.jpg'.format(track.track_id, self.frame_idx), box_vicinity * 255)

            thresh = threshold_otsu(box)
            box_dominant_pixels = box * (box > thresh)
            box_non_dominant_pixels = box * (box <= thresh)
            cv2.imwrite('/data/tkhurana/tk/deep_sort/verificationn/{}/{}_box_dominant_pixels.jpg'.format(track.track_id, self.frame_idx), box_dominant_pixels * 255)
            # box_vicinity_pixels = box_vicinity * (box_vicinity > thresh)
            cv2.imwrite('/data/tkhurana/tk/deep_sort/verificationn/{}/{}_box_non_dominant_pixels.jpg'.format(track.track_id, self.frame_idx), box_non_dominant_pixels * 255)
            box_dominant_mean = np.mean(box_dominant_pixels[np.nonzero(box_dominant_pixels)])
            box_non_dominant_mean = np.mean(box_non_dominant_pixels[np.nonzero(box_non_dominant_pixels)])

            if box_dominant_mean * noise > box_non_dominant_mean:
                previously_occluded_tracks.append(idx)
            else:
                unmatched_tracks.append(idx)


        return previously_occluded_tracks, unmatched_tracks

    def update_old(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections, newly_occluded_tracks, previously_occluded_tracks = \
            self._match(detections)

        # print(len(matches), len(unmatched_tracks),
        #       len(unmatched_detections), len(newly_occluded_tracks),
        #       len(previously_occluded_tracks))
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx], self.image, self.sequence_info)

        # if len(self.tracks) > 13:
        #     if self.tracks[13].track_id == 14:
        #         print(self.tracks[3].state)

        for track_idx in newly_occluded_tracks:
            self.tracks[track_idx].mark_occluded()

        for track_idx in previously_occluded_tracks:
            self.tracks[track_idx].mark_tentative()

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed() or t.is_occluded()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed() and not track.is_occluded():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)



    def _match_old(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed, occluded and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        occluded_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_occluded()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed() and not t.is_occluded()]

        # There are two things to note here:
        # (1) A TrackState.Occluded track will only emerge from a
        # TrackState.Confirmed track.
        # (2) However, for those tracks that were already TrackState.Occluded,
        # we should let the TrackState.Confirmed tracks match first and
        # TrackState.Occluded tracks match second, as a TrackState.Occluded
        # track that is recovering from occlusion would be less certain of
        # encountering a corresponding detection as compared to
        # TrackState.Confirmed.

        # (1) is implemented here.
        newly_occluded_tracks, confirmed_tracks = self.reason_for_occlusions(
                                                        self.tracks,
                                                        confirmed_tracks)


        newly_occluded_tracks = newly_occluded_tracks + occluded_tracks

        # if 4 in newly_occluded_tracks and self.tracks[4].track_id == 8:
        #     print("Track 8 is in the occluded state")
        # elif 4 in confirmed_tracks and self.tracks[4].track_id == 8:
        #     print("Track 8 is in the confirmed state")



        # if 4 in unmatched_tracks_a and self.tracks[4].track_id == 8:
        #     print("Track 8 was unmatched")
        # else:
        #      print("Track 8 was matched")

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, 0, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # (2) is implemented here.
        matches_c, newly_occluded_tracks, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, 0, self.max_age, # 0.15
                self.tracks, detections, newly_occluded_tracks, unmatched_detections)


        previously_occluded_tracks = []

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        # matches_d, newly_occluded_tracks, unmatched_detections = \
        #     linear_assignment.min_cost_matching(
        #         iou_matching.iou_cost, 0.9, self.tracks,
        #         detections, newly_occluded_tracks, unmatched_detections)

        # newly_occluded_tracks = newly_occluded_tracks + unmatched_occluded_tracks


        # if 4 in unmatched_tracks_b and self.tracks[4].track_id == 8:
        #     print("Track 8 was unmatched once again")
        # else:
        #      print("Track 8 was matched once again")

        matches = matches_a + matches_b + matches_c # + matches_d
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        # if 4 in unmatched_tracks and self.tracks[4].track_id == 8:
        #     print("Track 8 was unmatched finally")
        #     print(self.tracks[4].time_since_update)

        # previously_occluded_tracks, newly_occluded_tracks = self.reason_for_reappearances(
        #                                                 self.tracks,
        #                                                 newly_occluded_tracks)


        return matches, unmatched_tracks, unmatched_detections, newly_occluded_tracks, previously_occluded_tracks

