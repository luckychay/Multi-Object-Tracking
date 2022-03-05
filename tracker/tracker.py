'''
Description: 
Version: 
Author: Xuanying Chen
Date: 2022-03-03 21:18:35
LastEditTime: 2022-03-03 22:33:36
'''
from abc import abstractmethod

__all__ = ["Tracker"]

class Tracker(object):
    """
    Abstract base class for detectors.
    """
    def __init__(self) -> None:
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def update(self):
        """
        Subclasses must override this method, but adhere to the same return type.
        Returns:
            [x1,y1,x2,y2,track_id]
        """
        pass