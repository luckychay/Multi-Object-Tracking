'''
Description: 
Version: 
Author: Xuanying Chen
Date: 2022-02-16 17:09:59
LastEditTime: 2022-03-04 18:29:29
'''
import cv2
import numpy as np
from detector.yolov5.utils.augmentations import letterbox
from detector.yolov5.utils.datasets import LoadImages, LoadStreams

import rospy
from cm_transport.msg import CustomCImage
from sensor_msgs.msg import CompressedImage
from detection.msg import boxes,box

class LoadRosTopic:  # for inference
    # YOLOv5 rostopic dataloader, i.e. `python detect.py --source 1`
    def __init__(self, topic='/usb_cam/compressed', img_size=640, stride=32, auto=True, datatype=CompressedImage):
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.topic = topic
        self.datatype = datatype
        self.image = None
        rospy.init_node('detection_listener', anonymous=True)
        rospy.Subscriber(self.topic, datatype, self.callback, queue_size = 3)
        # rospy.spin()

    def callback(self,data):
        if self.datatype == CustomCImage:
            np_arr = np.fromstring(data.image.data, np.uint8)
            self.tram_status = data.tram_status.status
            rospy.loginfo(rospy.get_caller_id() + "I heard %d",self.tram_status)
        elif self.datatype == CompressedImage:
            np_arr = np.fromstring(data.data, np.uint8)
            self.tram_status = 0
        self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        
        self.count += 1
        # Read frame
        img0 = self.image.copy()
        
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride,auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return self.topic, img, img0, None, '' #self.tram_status      

    def __len__(self):
        return 0


class PublishRosTopic: 
    def __init__(self,topic='/detection/boxes',rate=10, datatype=boxes):
        self.rate = rospy.Rate(rate) 
        self.pub = rospy.Publisher(topic, datatype, queue_size=3)  
        # rospy.init_node('detection_talker', anonymous=True)
    
    def send(self,data):
        self.pub.publish(data)
        self.rate.sleep()


def handler(signum, frame):
    print("\nCtrl-c was pressed, exiting.")
    exit(1)