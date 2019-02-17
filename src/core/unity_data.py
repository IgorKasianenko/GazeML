import os
import numpy as np
import multiprocessing
from PIL import Image
import ujson

import util.gaze
import util.heatmap

class UnityData():
    """Simple Unity Eyes data loading class."""
    def __init__(self, 
                batch_size: int = 32, 
                unityeyes_path: str, 
                testing=False,
                generate_heatmaps=False,
                eye_image_shape=(36, 60),
                num_threads: int = max(4, multiprocessing.cpu_count()),
                heatmaps_scale=1.0 ):
        self._eye_image_shape = eye_image_shape
        self._heatmaps_scale = heatmaps_scale
        self._images_path = unityeyes_path



        # Define bounds for noise values for different augmentation types
        self._difficulty = 0.0
        self._augmentation_ranges = {  # (easy, hard)
            'translation': (2.0, 10.0),
            'rotation': (0.1, 2.0),
            'intensity': (0.5, 20.0),
            'blur': (0.1, 1.0),
            'scale': (0.01, 0.1),
            'rescale': (1.0, 0.2),
            'num_line': (0.0, 2.0),
            'heatmap_sigma': (5.0, 2.5),
        }
        self._generate_heatmaps = generate_heatmaps

        def load_eye(eye_path):
            jpg_path = '%s/%s.jpg' % (self._images_path, file_stem)
            json_path = '%s/%s.json' % (self._images_path, file_stem)
            with open(json_path, 'r') as f:
                json_data = ujson.load(f)
            entry = {
                'full_image': im = Image.open(jpg_path),
                'json_data': json_data
            }
            assert entry['full_image'] is not None
            return entry
        def entry_generator(self, yield_just_one=False):
            