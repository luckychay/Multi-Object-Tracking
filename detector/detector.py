'''
Description: 
Version: 
Author: Xuanying Chen
Date: 2022-03-03 21:18:35
LastEditTime: 2022-03-04 10:05:56
'''
from abc import abstractmethod
import torch.nn as nn

__all__ = ["Detector"]

class Detector(nn.Module): 
    """
    Abstract base class for detectors.
    """
    def __init__(self) -> None:
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self,im,augment=False, visualize=False, val=False):
        """
        Subclasses must override this method, but adhere to the same return type.
        Returns:
            [x,y,w,h,_id]
        """
        pass