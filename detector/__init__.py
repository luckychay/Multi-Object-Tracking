
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  #root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)+"/yolov5")  # add ROOT to PATH

from .build import build_detector, DETECTOR_REGISTRY  
from .detector import Detector


__all__ = [k for k in globals().keys() if not k.startswith("_")]
# TODO can expose more resnet blocks after careful consideration
