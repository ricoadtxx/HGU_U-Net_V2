import cv2
import numpy as np
import albumentations as A

class ResampleTransform(A.ImageOnlyTransform):
    def __init__(self, scale_limit=0.5, interpolation=cv2.INTER_NEAREST, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.scale_limit = scale_limit
        self.interpolation = interpolation

    def apply(self, image, **params):
        h, w = image.shape[:2]
        scale = np.random.uniform(self.scale_limit, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        resampled = cv2.resize(image, (new_w, new_h), interpolation=self.interpolation)
        resampled = cv2.resize(resampled, (w, h), interpolation=self.interpolation)
        return resampled
