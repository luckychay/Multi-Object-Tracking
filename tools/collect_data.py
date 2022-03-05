#ROS
import rospy
import os
import numpy as np
import cv2
import time
import argparse
from sensor_msgs.msg import CompressedImage

VERBOSE = True

class Listener:
    def __init__(self,topic):
        
        rospy.Subscriber(topic, CompressedImage, self.callback)
        if VERBOSE:
            print("subscriber is created.")
        
    def callback(self,data):
        np_arr = np.fromstring(data.data, np.uint8)
        self.image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if VERBOSE:
            cv2.imshow("collecting data",self.image_np)
            cv2.waitKey(1)
        img_name = "./raw/"+time.strftime("%Y%m%d%H%M%S",time.localtime())+".jpg"
        if not os.path.exists(img_name):
            cv2.imwrite(img_name,self.image_np)
            print(img_name," saved.")

def main(opt):
    topic = opt.rostopic    
    if VERBOSE:
        print("topic:",opt.rostopic)

    img_getter = Listener(topic)
    rospy.init_node('listener', anonymous=True)
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rostopic', type=str, default='/rgb_cam/image_raw/compressed', help='source')
    opt = parser.parse_args()

    main(opt)