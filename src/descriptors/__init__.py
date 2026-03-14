from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray

from src.utils.config import Config