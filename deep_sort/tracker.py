# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import os
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from pycocotools import mask as maskUtils
import cv2
from skimage.transform import resize


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
        self.only_filtering = False
        self.temporal_noise = True
        self.ah_velocity = False
        self.velocity_weighting = True
        self.tn = -1
        self.motion_aware = False
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
            bboxes.append(track.to_tlwh)

        impath = os.path.join(
                    self.depth_map_path,
                    'img1',
                    '{:06d}.jpg'.format(self.frame_idx))

        self.masks = get_mask_from_bbox(bboxes, impath)

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf, self.max_height, tn=self.tn,
                          warp_matrix=self.warp_matrix)

    def update(self, detections, occluded_factor=1.0,
               filtering_factor=1.0, tn=self.tn):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections, newly_occluded_tracks, previously_occluded_tracks = \
            self._match(detections, occluded_factor=occluded_factor, filtering_factor=filtering_factor)

        # for all the matched detection and track pairs, we are going to (conditionally) call
        # these confirmed tracks and do the needful (as you can find in the update function in
        # track.py in this folder).
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx],
                self.image, self.sequence_info)

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

    def _match(self, detections, occluded_factor=1.0, filtering_factor=1.0):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices, tn=self.tn)

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
        newly_occluded_tracks, confirmed_tracks = self.reason_for_occlusions(
                                                    self.tracks,
                                                    confirmed_tracks,
                                                    occluded_factor)
        newly_occluded_tracks = newly_occluded_tracks + occluded_tracks

        # if using default matching, merge all kinds of tracks together into confirmed_tracks
        # and match these together based on appearance. later we will segregate them again
        confirmed_tracks = confirmed_tracks + newly_occluded_tracks
        matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    gated_metric, self.metric.matching_threshold, 0, self.max_age,
                    self.tracks, detections, confirmed_tracks)

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

        matches = matches_a + matches_b

        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        # this step segregates the occluded tracks from the unmatched confirmed tracks
        # if you used default matching above, (because we merged both into one for default
        # matching)
        newly_occluded_tracks = [i for i in newly_occluded_tracks if i in unmatched_tracks]
        unmatched_tracks = [i for i in unmatched_tracks if i not in newly_occluded_tracks]

        pv1, occluded_tracks_ = self.reason_for_reappearances(
                                                    self.tracks,
                                                    newly_occluded_tracks,
                                                    filtering_factor)
        pv2, unmatched_tracks = self.reason_for_reappearances(
                                                    self.tracks,
                                                    unmatched_tracks,
                                                    filtering_factor)
        previously_occluded_tracks = pv1 + pv2

        return matches, unmatched_tracks, unmatched_detections, occluded_tracks_, previously_occluded_tracks

    def _initiate_track(self, detection, temporal_noise=True, tn=-1):
        mean_depth = self.compute_mean_depth(self.image, detection, self.sequence_info)
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
            m = maskUtils.decode(detection.mask.copy())
        elif mask is not None:
            m = maskUtils.decode(mask)
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

        return warp_matrix

    def update_metadata(self, idx, path, seq_info, max_height, tn=-1):
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
        self.tn = tn
        if self.frame_idx != 1:
            self.warp_matrix = self.align(self.past_frame,
                                      self.current_frame)

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
        scale_x = self.sequence_info["image_size"][1] / float(image.shape[1])
        scale_y = self.sequence_info["image_size"][0] / float(image.shape[0])

        for idx in track_indices:
            track = self.tracks[idx]
            _, _, _, _, predicted_depth = track.to_tlwhz()

            box_mean = compute_mean_depth_from_mask(
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
        scale_x = self.sequence_info["image_size"][1] / float(image.shape[1])
        scale_y = self.sequence_info["image_size"][0] / float(image.shape[0])

        for idx in track_indices:
            track = self.tracks[idx]
            _, _, _, _, predicted_depth = track.to_tlwhz()

            box_mean = compute_mean_depth_from_mask(
                    image, None, self.sequence_info, self.masks[idx])

            if predicted_depth > box_mean * filtering_factor:
                previously_occluded_tracks.append(idx)
            else:
                occluded_tracks.append(idx)

        return previously_occluded_tracks, occluded_tracks

