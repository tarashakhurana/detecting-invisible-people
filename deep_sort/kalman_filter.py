# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg
import cv2
import time


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 10-dimensional state space

        x, y, a, h, z, vx, vy, va, vh, vz

    contains the bounding box center position (x, y), aspect ratio a, height h,
    depth z from a monocular depth estimator and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h, z) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 5, 1.
        self.ndim = ndim
        self.dt = dt

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement, tn=-1):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h, z) with center position (x, y),
            aspect ratio a, and height h, and depth z.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (10 dimensional) and covariance matrix (10x10
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        if tn==-1:
            std = [
                    2 * self._std_weight_position * 500,
                    2 * self._std_weight_position * 500,
                    1e-2,
                    2 * self._std_weight_position * 500,
                    1e-2,
                    10 * self._std_weight_velocity * 500,
                    10 * self._std_weight_velocity * 500,
                    1e-5,
                    10 * self._std_weight_velocity * 500,
                    0.0075]
        else:
            std = [
                    tn['of'] * 2 * self._std_weight_position * tn['oc'],
                    tn['of'] * 2 * self._std_weight_position * tn['oc'],
                    tn['of'] * 1e-2,
                    tn['of'] * 2 * self._std_weight_position * tn['oc'],
                    tn['of'] * 1e-2,
                    tn['of'] * 10 * self._std_weight_velocity * tn['oc'],
                    tn['of'] * 10 * self._std_weight_velocity * tn['oc'],
                    tn['of'] * 1e-5,
                    tn['of'] * 10 * self._std_weight_velocity * tn['oc'],
                    tn['of'] * 0.0075]

        covariance = np.diag(np.square(std))
        return mean, covariance

    def make_pos(self, cx, cy, ar, height):
        width = ar * height
        return [[cx - width / 2,
                cy - height / 2,
                cx + width / 2,
                cy + height / 2]]

    def make_xyah(self, pos):
        x = (pos[0][0] + pos[0][2]) / 2
        y = (pos[0][1] + pos[0][3]) / 2
        w = pos[0][2] - pos[0][0]
        h = pos[0][3] - pos[0][1]
        a = float(w) / h
        return x, y, a, h

    def warp_pos(self, mean, warp_matrix):
        pos = self.make_pos(mean[0], mean[1], mean[2], mean[3])
        p1 = np.array([pos[0][0], pos[0][1], 1]).reshape((3, 1))
        p2 = np.array([pos[0][2], pos[0][3], 1]).reshape((3, 1))
        p1_n = np.matmul(warp_matrix, p1).reshape((1, 2))
        p2_n = np.matmul(warp_matrix, p2).reshape((1, 2))
        pos = np.concatenate((p1_n, p2_n), 1).reshape((1, -1))
        return np.concatenate((self.make_xyah(pos), mean[4:]))

    def predict(self, mean, covariance, motion_mat=[0],
                tn=-1, warp_matrix=[]):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 10 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 10x10 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        if len(motion_mat) == 1:
            motion_mat = self._motion_mat

        if type(warp_matrix) != int:
            start = time.time()
            mean = self.warp_pos(mean, warp_matrix)

        if tn==-1:
            std_pos = [
                    self._std_weight_position * 500 * mean[4],
                    self._std_weight_position * 500 * mean[4],
                    1e-2 * mean[4],
                    self._std_weight_position * 500 * mean[4],
                    1e-2]
            std_vel = [
                    self._std_weight_velocity * 500 * mean[4],
                    self._std_weight_velocity * 500 * mean[4],
                    1e-5 * mean[4],
                    self._std_weight_velocity * 500 * mean[4],
                    1e-5]
        else:
            std_pos = [
                    tn['pf'] * self._std_weight_position * tn['pc'] * mean[4],
                    tn['pf'] * self._std_weight_position * tn['pc'] * mean[4],
                    tn['pf'] * 1e-2 * mean[4],
                    tn['pf'] * self._std_weight_position * tn['pc'] * mean[4],
                    tn['pf'] * 1e-2]
            std_vel = [
                    tn['pf'] * self._std_weight_velocity * tn['pc'] * mean[4],
                    tn['pf'] * self._std_weight_velocity * tn['pc'] * mean[4],
                    tn['pf'] * 1e-5 * mean[4],
                    tn['pf'] * self._std_weight_velocity * tn['pc'] * mean[4],
                    tn['pf'] * 1e-5]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(motion_mat, mean)
        covariance = np.linalg.multi_dot((
            motion_mat, covariance, motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, tn=-1):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (10 dimensional array).
        covariance : ndarray
            The state's covariance matrix (10x10 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        if tn==-1:
            std = [
                    self._std_weight_position * 500,
                    self._std_weight_position * 500,
                    1e-1,
                    self._std_weight_position * 500,
                    1e-1]
        else:
            std = [
                    tn['of'] * self._std_weight_position * tn['oc'],
                    tn['of'] * self._std_weight_position * tn['oc'],
                    tn['of'] * 1e-1,
                    tn['of'] * self._std_weight_position * tn['oc'],
                    tn['of'] * 1e-1]

        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement, tn=-1):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (10 dimensional).
        covariance : ndarray
            The state's covariance matrix (10x10 dimensional).
        measurement : ndarray
            The 5 dimensional measurement vector (x, y, a, h, z), where (x, y)
            is the center position, a the aspect ratio, and h the height, and
            z the depth of the bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance,
                                                    tn=tn)

        if measurement[-1] == -1:
            measurement[-1] = projected_mean[-1]

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, tn=-1):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance, tn=tn)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        mean = mean[:4]
        covariance = covariance[:4, :4]
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

    def get_ndim(self):
        return self.ndim

    def get_dt(self):
        return self.dt
