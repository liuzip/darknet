#!/bin/sh

../darknet detector demo -s 24 -dont_show data/ibm.data cfg/yolo-ibm.cfg weights/yolo-ibm_40000.weights ../long_video.mp4
#../darknet detector demo -s 24 -dont_show data/face.data cfg/yolo-face.cfg weights/yolo-face_500000.weights ../output.avi
