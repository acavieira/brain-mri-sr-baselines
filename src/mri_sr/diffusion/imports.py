# train
import csv
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from ..io import ensure_dir
from .config import DiffusionConfig
from .dataset import NiftiSlicePairDataset, collect_nifti_paths
from .model import ConditionalUNet2D
from .scheduler import DiffusionScheduler

#scheduler
from dataclasses import dataclass

# dataset
import glob
import cv2
from ..degradation import degrade_image
from ..io import choose_slice_indices, extract_slice, load_nifti_volume
from ..preprocessing import prepare_hr_reference

# model
import math
import torch.nn as nn


 
