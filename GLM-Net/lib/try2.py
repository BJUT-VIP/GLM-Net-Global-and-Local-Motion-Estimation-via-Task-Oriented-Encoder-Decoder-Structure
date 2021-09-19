import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2


if __name__ == '__main__':
    ref = np.arange(0, 100, 1).astype(np.uint8)
    ref = np.repeat(ref.reshape(1, 100), 100, axis=0)

    ref1 = cv2.resize(ref, (3, 3))

    pass

