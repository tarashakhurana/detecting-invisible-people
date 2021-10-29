# vim: expandtab:ts=4:sw=4
import numpy as np
from scipy.interpolate import CubicSpline
from pycocotools import mask as maskUtils
from skimage.transform import resize


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks. The state of a `confirmed` track can be changed to `occluded` as soon
    as there is an impending occlusion in the scene for that track.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Occluded = 4


class Track:
    """
    A single target track with state space `(x, y, a, h, z)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height and `z` is the depth.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.track_history = []

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlwhz(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height, depth)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:5].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:4] / 2
        return ret

    def to_tlwhz_cov(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height, depth, covariance in xz)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:5].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:4] / 2

        xx = self.covariance[0][0]
        xz = self.covariance[0][4]
        zx = self.covariance[4][0]
        zz = self.covariance[4][4]
        return list(ret) + [xx, xz, zx, zz]

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf, max_height, update_age=True, tn=-1, warp_matrix=[]):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        if self.state == TrackState.Occluded:
            h = self.to_tlwh()[3]
            ndim = kf.get_ndim()
            dt = kf.get_dt()
            motion_mat = np.eye(2 * ndim, 2 * ndim)
            for i in range(ndim):
                motion_mat[i, ndim + i] = 1
            motion_mat[3, -2] = 0 #h / float(max_height)
            motion_mat[2, -3] = 0 #h / float(max_height)
            motion_mat[4, -1] = 1
            if update_age:
                self.mean, self.covariance = kf.predict(self.mean,
                                self.covariance, motion_mat, tn=tn,
                                warp_matrix=warp_matrix)
            else:
                return kf.predict(self.mean, self.covariance,
                                motion_mat, tn=tn,
                                warp_matrix=warp_matrix)
        else:
            if update_age:
                self.mean, self.covariance = kf.predict(self.mean, self.covariance,
                                                tn=tn, warp_matrix=warp_matrix)
            else:
                return kf.predict(self.mean, self.covariance,
                                  tn=tn, warp_matrix=warp_matrix)
        if update_age:
            self.age += 1
            self.time_since_update += 1


    def update(self, kf, detection, depth_map, seq_info, tn=-1):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.track_history.append(detection.to_xyah())
        if len(self.track_history) > 20:
            self.track_history = self.track_history[1:]
        mean_depth = self.compute_mean_depth(depth_map, detection, seq_info)
        det = list(detection.to_xyah())
        det = det + [mean_depth]
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, det, tn=tn)
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0

        # If a track was occluded and it was matched with a detection, then
        # update its state to TrackState.Confirmed. Checking hits against
        # self._n_init for an occluded state is not required as this was a
        # TrackState.Confirmed track when it entered TrackState.Occluded.
        if self.state == TrackState.Occluded:
            self.state = TrackState.Confirmed
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def compute_mean_depth(self, depth_map, detection, seq_info):
        scale_x = seq_info["image_size"][1] / float(depth_map.shape[1])
        scale_y = seq_info["image_size"][0] / float(depth_map.shape[0])
        box = detection.tlwh.copy()
        box[2:] += box[:2]
        box = [box[0]/scale_x, box[1]/scale_y, box[2]/scale_x, box[3]/scale_y]
        box = [int(x) for x in box]
        box = [max(0, box[0]), max(0, box[1]),
               max(0, min(depth_map.shape[1], box[2])),
               max(0, min(depth_map.shape[0], box[3]))]

        if 0 in box[2:] or box[0] >= depth_map.shape[1] or box[1] >= depth_map.shape[0] or box[0] == box[2] or box[1] == box[3]:
            return -1

        box = depth_map[box[1]:box[3], box[0]:box[2]].copy()
        return np.mean(box)

    def compute_mean_depth_from_mask(self, depth_map, detection, seq_info):
        width = depth_map.shape[1]
        height = depth_map.shape[0]

        mask = maskUtils.decode(detection.mask.copy())
        mask = resize(mask, (height, width), order=1)

        inter_mask = np.zeros((height, width), dtype=float)
        inter_mask = np.where(mask > 10e-6, depth_map, 0)

        if 0 in np.nonzero(inter_mask)[0].shape:
            return -1
        return np.mean(inter_mask[np.nonzero(inter_mask)])


    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted

        # The following statement automatically handles the TrackState.Occluded
        # tracks. Deletes them if occlusion lasts for more than self._max_age.
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def mark_deleted(self):
        """Mark this track as missed (no association at the current time step).
        """
        self.state = TrackState.Deleted

    def mark_occluded(self):
        """Mark this track as occluded (no association at the current time step).
        """
        if self.state == TrackState.Confirmed:
            self.state = TrackState.Occluded
        if self.state == TrackState.Occluded and self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_occluded(self):
        """Returns True if this track is occluded."""
        return self.state == TrackState.Occluded

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
