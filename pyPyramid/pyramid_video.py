import numpy as np
import cv2


class PyramidVideo:
    def __init__(self, video: np.ndarray, window_sizes: list, levels_used: list, factor: int = 2):
        """
        Builds a Gaussian image pyramid over a video and organises the levels
        requested by the caller.

        Parameters
        ----------
        video : np.ndarray
            Input video array with shape (n_frames, height, width).
        window_sizes : list
            Window (ROI) size for each pyramid level.  Index corresponds to the
            absolute pyramid level (0 = original resolution, 1 = half, …).
        levels_used : list[int]
            Pyramid levels to use, given in the order they will be applied
            during DIC (typically from coarsest to finest, e.g. [4, 2, 1, 0]).
        factor : int, optional
            Down-sampling factor between consecutive levels (default 2).
        """
        self.video = video
        self.factor = factor
        self.window_sizes_all = window_sizes   # indexed by absolute level
        self.levels_used = levels_used
        self.max_level = max(levels_used)
        self.scales_all = np.array([self.factor ** i for i in range(self.max_level + 1)])

        self._build_pyramid()
        self._sort_levels()

    # ------------------------------------------------------------------
    def _build_pyramid(self):
        """Builds the full pyramid up to max_level."""
        self._pyramid_all = [self.video]
        for _ in range(self.max_level):
            downsampled = np.array([cv2.pyrDown(frame) for frame in self._pyramid_all[-1]])
            self._pyramid_all.append(downsampled)

    # ------------------------------------------------------------------
    def _sort_levels(self):
        """
        Reorders pyramid data so that index 0 corresponds to levels_used[0],
        index 1 to levels_used[1], etc.
        """
        n = len(self.levels_used)
        self.video_pyramid = [None] * n
        self.scale        = np.zeros(n, dtype=np.int32)
        self.window_sizes = np.zeros(n, dtype=np.int32)

        for i, lvl in enumerate(self.levels_used):
            self.video_pyramid[i] = self._pyramid_all[lvl]
            self.scale[i]         = self.scales_all[lvl]
            self.window_sizes[i]  = self.window_sizes_all[lvl]
