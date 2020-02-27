# Autonomous Parking Attendant

## Summary
Using Python and ROS, created a simulated autonomous vehicle capable of navigating a model village, reading parked car license plates, and responding to objects such as moving vehicles, pedestrians, and crosswalks.

## Approach
Implemented 2 ROS nodes: an Object Recognition (OR) node and a Driver node. I was primarily responsible for the OR node, which:

* Used a convolutional neural network, developed with TensorFlow and Keras, to recognize and interpret license plates. 
* Processed the vehicle’s visual feed with OpenCV; using masking, bounding, and geometric transformations for hazard recognition and preparation of images for the neural network.
