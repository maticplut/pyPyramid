import time

import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline
from tqdm.auto import tqdm

from .pyramid_video import PyramidVideo


# ======================================================================
# Low-level DIC helpers
# ======================================================================

def jacobijan(height, width):
    zeros = np.zeros((height, width), dtype=np.float64)
    ones  = np.ones((height, width),  dtype=np.float64)
    cx = width  // 2.0
    cy = height // 2.0
    x, y = np.meshgrid(np.arange(width)  - cx,
                       np.arange(height) - cy)
    return np.array([[zeros,  ones, -y],
                     [ones, zeros,   x]])


def rigid_transform_matrix(p):
    return np.array([[np.cos(p[2]), -np.sin(p[2]), p[1]],
                     [np.sin(p[2]),  np.cos(p[2]), p[0]],
                     [0,             0,             1  ]])


def transform_params(matrix):
    tx  = matrix[0, 2]
    ty  = matrix[1, 2]
    phi = np.arctan2(-matrix[1, 0], matrix[0, 0])
    return np.array([ty, tx, phi])


def get_roi_image(image, roi_center, roi_size):
    """Extract a square ROI centred on roi_center = (row, col)."""
    y = np.arange(roi_center[1] - roi_size // 2,
                  roi_center[1] + (roi_size - 1) // 2).astype(int)
    x = np.arange(roi_center[0] - roi_size // 2,
                  roi_center[0] + (roi_size - 1) // 2).astype(int)
    X, Y = np.meshgrid(x, y)
    return image[Y, X]


def coordinate_warp(warp, roi_size):
    h, w = roi_size
    x = np.tile(np.arange(w), h) - w // 2
    y = np.repeat(np.arange(h), w) - h // 2
    coords  = np.vstack((x, y, np.ones_like(x)))
    warped  = warp @ coords
    xi = warped[0] + w // 2
    yi = warped[1] + h // 2
    return xi, yi


def interpolate_image(xi, yi, image, output_shape, spl=None):
    h, w = output_shape
    if spl is None:
        spl = RectBivariateSpline(np.arange(image.shape[0]),
                                  np.arange(image.shape[1]),
                                  image)
    return spl.ev(yi, xi).reshape((h, w))


def get_gradient(image):
    gx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    return np.array([gx, gy])


def steepest_descent_image(jacobian, gradient):
    """
    Parameters
    ----------
    jacobian : ndarray, shape (2, 3, H, W)
    gradient : ndarray, shape (2, H, W)  — normalised image gradient
    """
    jacx, jacy = jacobian
    gx, gy     = gradient
    return jacx * gx + jacy * gy


def hessian(sd_image):
    H = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            H[i, j] = np.sum(sd_image[i] * sd_image[j])
    return H


def error_vector(error_image, sd_image):
    b = np.zeros(3, dtype=np.float64)
    for i in range(3):
        b[i] = np.sum(sd_image[i] * error_image)
    return b


def error_image_ZNSSD(reference, current):
    f_ = np.mean(reference);  sd_f = np.std(reference)
    g_ = np.mean(current);    sd_g = np.std(current)
    return ((reference - f_) / sd_f - (current - g_) / sd_g).astype(np.float64)


def error_image_ZNCC(reference, current):
    f_ = np.mean(reference);  sd_f = np.std(reference)
    g_ = np.mean(current);    sd_g = np.std(current)
    return ((reference - f_) - sd_f / sd_g * (current - g_)).astype(np.float64)


def error_image_SSD(reference, current):
    return (reference.astype(np.float64) - current.astype(np.float64))


# ======================================================================
# Main pyramid DIC function
# ======================================================================

def rigid_dic_pyramid_displacement(
    pyramid: PyramidVideo,
    start_position,
    n_frame_skip: int = 1,
):
    """
    Rigid-body DIC with image-pyramid acceleration.

    The algorithm follows the inverse-compositional Lucas–Kanade scheme with
    the ZNSSD criterion.  At each frame the warp is first estimated at the
    coarsest pyramid level, then refined at each successive finer level,
    using the previous level's result as the starting guess.

    Parameters
    ----------
    pyramid : PyramidVideo
        Pre-built pyramid object.  Its ``levels_used`` attribute determines
        which pyramid levels are visited (in order, coarsest first).
    start_position : array-like, shape (2,)
        (row, col) position of the ROI centre in the *original* (level-0)
        frame.
    n_frame_skip : int
        Process every n-th frame (default 1 = every frame).

    Returns
    -------
    results : ndarray, shape (n_frames, n_levels, 3)
        Estimated parameters [dy, dx, dtheta] for every processed frame and
        every pyramid level, scaled back to original-image pixel units.
    iterations : ndarray, shape (n_frames, n_levels)
        Number of IC-LK iterations per frame per level.
    times : ndarray, shape (n_frames, n_levels)
        Wall-clock seconds per frame per level.
    """
    start_position = np.asarray(start_position, dtype=np.float64)
    n_levels = len(pyramid.levels_used)

    # ------------------------------------------------------------------
    # Pre-compute level-constant quantities (Jacobian, SD image, Hessian)
    # ------------------------------------------------------------------
    roi_references = []
    sd_images      = []
    inv_hessians   = []

    for i in range(n_levels):
        reference     = pyramid.video_pyramid[i][0]
        scale_i       = pyramid.scale[i]
        window_size_i = pyramid.window_sizes[i]

        roi_ref = get_roi_image(reference,
                                start_position / scale_i,
                                window_size_i)

        jac            = jacobijan(*roi_ref.shape)
        grad           = get_gradient(roi_ref)
        grad_norm      = grad / np.std(roi_ref)        # ZNSSD normalisation
        sd_img         = steepest_descent_image(jac, grad_norm)
        inv_hessians.append(np.linalg.inv(hessian(sd_img)))
        roi_references.append(roi_ref)
        sd_images.append(sd_img)

    # ------------------------------------------------------------------
    # Per-frame optimisation
    # ------------------------------------------------------------------
    all_results    = np.empty((0, 3), dtype=np.float64)
    all_iterations = np.array([], dtype=int)
    all_times      = np.array([], dtype=float)

    p    = np.zeros(3, dtype=np.float64)
    warp = rigid_transform_matrix(p)

    frame_indices = range(0, len(pyramid.video_pyramid[0]), n_frame_skip)

    for frame_idx in tqdm(frame_indices, desc='Frames', ncols=None):

        iter_per_level = np.array([], dtype=int)
        time_per_level = np.array([], dtype=float)

        for i in range(n_levels):
            t0 = time.time()

            current_frame = pyramid.video_pyramid[i][frame_idx]
            roi_ref       = roi_references[i]
            sd_img        = sd_images[i]
            inv_H         = inv_hessians[i]
            scale_i       = float(pyramid.scale[i])
            window_size_i = int(pyramid.window_sizes[i])

            h, w = current_frame.shape
            spl  = RectBivariateSpline(np.arange(h), np.arange(w), current_frame)

            # Scale translation components to current pyramid level
            warp[0, 2] /= scale_i
            warp[1, 2] /= scale_i

            n_iter = 0
            err    = 1.0

            while err > 1e-5 and n_iter < 4000:
                xi, yi = coordinate_warp(warp, roi_ref.shape)

                # Shift coordinates to the ROI centre in the current frame
                xi += start_position[0] / scale_i - window_size_i // 2
                yi += start_position[1] / scale_i - window_size_i // 2

                warped_roi  = interpolate_image(xi, yi, current_frame,
                                                roi_ref.shape, spl=spl)
                err_img     = error_image_ZNSSD(roi_ref, warped_roi)
                b           = error_vector(err_img, sd_img)
                dp          = -(inv_H @ b)
                err         = np.linalg.norm(dp)
                dp_warp     = rigid_transform_matrix(dp)
                warp        = warp @ np.linalg.inv(dp_warp)
                p           = transform_params(warp)
                n_iter     += 1

            # Scale translation back to original resolution
            warp[0, 2] *= scale_i
            warp[1, 2] *= scale_i

            iter_per_level = np.append(iter_per_level, n_iter)
            time_per_level = np.append(time_per_level, time.time() - t0)

            # Store result scaled to original-image units
            all_results = np.vstack([all_results,
                                     p * np.array([scale_i, scale_i, 1.0])])

        all_iterations = np.append(all_iterations, iter_per_level)
        all_times      = np.append(all_times,      time_per_level)

    n_frames = len(frame_indices)
    results    = all_results.reshape(n_frames, n_levels, 3)
    iterations = all_iterations.reshape(n_frames, n_levels)
    times      = all_times.reshape(n_frames, n_levels)

    return results, iterations, times
