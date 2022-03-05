'''
Description: 
Version: 
Author: Xuanying Chen
Date: 2022-03-03 21:17:05
LastEditTime: 2022-03-04 10:59:39
'''
from .detector import Detector
from tools.registry import Registry

DETECTOR_REGISTRY = Registry("DETECTOR")
DETECTOR_REGISTRY.__doc__ = """
Registry for detectors, which tracks objects based on detectors
The registered object must be a callable that accepts one arguments:
args
Registered object must return instance of :class:`DETECTOR`.
"""

def build_detector(args):
    detector_name = args.detector
    detector = DETECTOR_REGISTRY.get(detector_name)(args)
    assert isinstance(detector,Detector)
    return detector
