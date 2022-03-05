'''
Description: 
Version: 
Author: Xuanying Chen
Date: 2022-03-03 21:17:05
LastEditTime: 2022-03-04 10:59:46
'''
from .tracker import Tracker
from tools.registry import Registry

TRACKER_REGISTRY = Registry("TRACKER")
TRACKER_REGISTRY.__doc__ = """
Registry for trackers, which tracks objects based on detectors
The registered object must be a callable that accepts one arguments:
args
Registered object must return instance of :class:`Tracker`.
"""

def build_tracker(args):
    tracker_name = args.tracker
    tracker = TRACKER_REGISTRY.get(tracker_name)(args)
    assert isinstance(tracker,Tracker)
    return tracker
