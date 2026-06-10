import numpy as np
import cv2


class PyramidVideo:
    def __init__(self, video: np.ndarray, levels_used: list, window_sizes: list = None, default_window_size: int = 50, factor: int = 2):
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
            Pyramid levels to use during DIC (typically from coarsest to finest, e.g. [4, 2, 1, 0]).
        factor : int, optional
            Down-sampling factor between consecutive levels (default 2).
        """
        self.video = video
        self.factor = factor
        self.window_sizes_all = window_sizes   # indexed by absolute level [level0, level1, ...]
        self.levels_used = levels_used
        self.max_level = max(levels_used)

        self.calculateShapes()

        if self.window_sizes_all is None:
            self.calculateWindowSizes(default_window_size)

        self.buildVideoPyramids()
        self.sortPyramidLevels()

    # ------------------------------------------------------------------
    def calculateShapes(self):
        '''Calculates the shapes of all pyramid levels up to max_level. Also calculates the scales for each level.'''
        shapes = []
        scales = []
        for i in range(self.max_level+1):
            scale = self.factor**i
            new_shape = (self.video.shape[0], int(self.video.shape[1] / scale), int(self.video.shape[2] / scale))
            shapes.append(new_shape)
            real_scale = self.video.shape[1] / new_shape[1]
            scales.append(real_scale)
        self.shapes_all = shapes
        self.scales_all = scales


    # ------------------------------------------------------------------
    def buildVideoPyramids(self):
        """Builds the full pyramid up to max_level."""
        
        self.pyramids_all = [self.video] # indexed by absolute level
        for _ in range(1, self.max_level+1):
            downsampled = np.array([cv2.pyrDown(frame).astype(np.uint8) for frame in self.pyramids_all[-1]])
            self.pyramids_all.append(downsampled)

    # ------------------------------------------------------------------
    def buildVideoPyramids_test(self):
        """Builds the full pyramid up to max_level."""
        
        self.pyramids_all = [self.video] # indexed by absolute level
        for i in range(1, self.max_level+1):
            video_to_downsample = self.pyramids_all[-1]
            downsampled = np.array([cv2.resize(cv2.GaussianBlur(frame, (5, 5), 0),
                                                self.shapes_all[i][1:],
                                                interpolation=cv2.INTER_AREA).astype(np.uint8) for frame in video_to_downsample]
                                    )
            self.pyramids_all.append(downsampled)

    # ------------------------------------------------------------------
    def sortPyramidLevels(self):
        """
        Reorders pyramid data so that index 0 corresponds to levels_used[0],
        index 1 to levels_used[1], etc.
        """
        # sort levels_used in descending order
        self.levels_used = sorted(self.levels_used, reverse=True)

        # pre-allocate arrays for the requested levels
        n = len(self.levels_used)
        self.video_pyramids = [None] * n
        self.scales = np.zeros(n, dtype=np.float32)
        self.window_sizes = np.zeros(n, dtype=np.int32)
        self.shapes = [None] * n 

        for i, lvl in enumerate(self.levels_used):
            self.video_pyramids[i] = self.pyramids_all[lvl]
            self.scales[i] = self.scales_all[lvl]
            self.window_sizes[i] = self.window_sizes_all[lvl]
            self.shapes[i] = self.shapes_all[lvl]

    # ------------------------------------------------------------------
    def calculateWindowSizes(self, base_window_size):
        '''Calculates window sizes for each level based on the base window size and the scales.'''
        self.window_sizes_all = []

        for i, scale in enumerate(self.scales_all):
            window_size_i = int(base_window_size // (scale * (0.75 ** i)))
            if window_size_i % 2 == 0:
                window_size_i += 1
            self.window_sizes_all.append(window_size_i)
