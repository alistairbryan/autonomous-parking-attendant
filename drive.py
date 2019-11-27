#!/usr/bin/env python
import random
import pickle
import cv2 as cv
import math
import rospy
import roslaunch
import time
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import subprocess
from sensor_msgs.msg import Image
from time import sleep
#GET Q VALUE FOR STATE


def getQ(self, state, action):
	return self.q.get((state, action), 0.0)		


def getState(self, data):
	#WILL RETURN THIS
	state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	#CONVERT IMAGE TO CV FORMAT
	try:
		im = self.bridge.imgmsg_to_cv2(data, "bgr8")
	except CvBridgeError as e:
		print(e)
	state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	#PREPARE IMAGE TO USE
	y = im.shape[0]
	x = im.shape[1]
	imageUn = im[(y-50):y,0:x]
	y = imageUn.shape[1]
	sec = x/10
	#CHECK FOR CROSSWALK RED
	crossPx = cv.countNonZero(cv.inRange(cv.cvtColor(imageUn,cv.COLOR_BGR2HSV), self.lower_red,self.upper_red ))
	if (crossPx > 20):
		if self.notCrossWalk == 0: #if we have seen crosswalk
			if self.crossWalk == 0:  
				self.crossWalk = 0
				self.notCrossWalk = 1
	else:
		if self.notCrossWalk == 1:
			self.notCrossWalk = 0
	#SET STATES
	for i in range(0,9):
		section = imageUn[0:y, sec*i:sec*(i + 1)]

		#ROAD
		roadPx = cv.countNonZero(cv.inRange(cv.cvtColor(section,cv.COLOR_BGR2HSV), self.lower_road,self.upper_road ))		
		#EVERYTHING THAT ISN'T ROAD
		carPx = cv.countNonZero(cv.inRange(cv.cvtColor(section,cv.COLOR_BGR2HSV), self.lower_car,self.upper_car ))
		grassPx = cv.countNonZero(cv.inRange(cv.cvtColor(section,cv.COLOR_BGR2HSV), self.lower_grass,self.upper_grass ))
		whitePx = cv.countNonZero(cv.inRange(cv.cvtColor(section,cv.COLOR_BGR2HSV), self.lower_white,self.upper_white ))*self.crossWalk
		notRoadPx = 0.8*float(carPx + grassPx + whitePx)

		if carPx > 10: #CAR IN COLUMN: 2
			state[i] = 2
		elif roadPx < notRoadPx: #NOT ROAD: 0
			state[i] = 0
		else: #ROAD : 1
			state[i] = 1
	print(state)
	return state



class Drive(gazebo_env.GazeboEnv):
	#LOAD Q VALUES

	def __init__(self):
		print("Started")
		self.imSub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
		self.vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)	
		self.bridge = CvBridge()
		#COLOR FILTER RANGES
		self.lower_red = np.array([0,  255,   255])
		self.upper_red = np.array([0,  255,   255])
		self.lower_blue = np.array([0,  0,   250])
		self.upper_blue = np.array([255, 255, 255])
		self.upper_car = np.array([255, 255, 210])
		self.lower_car = np.array([100, 0, 150])
		self.lower_grass = np.array([0, 0, 32])
		self.upper_grass = np.array([255, 255 ,180])
		self.lower_white = np.array([0, 0, 200])
		self.upper_white = np.array([200, 100 ,255])
		self.lower_road = np.array([0,0, 60])
		self.upper_road = np.array([0, 0, 90])


		self.action_space = spaces.Discrete(3)  # F,L,R
		self.actions =  range(self.action_space.n)

		self.crossWalk = 1
		self.notCrossWalk = 0


		self.q ={}
		with open("/home/chloe/QValues") as f: self.q = pickle.load(f)


	def callback(self, data):
		#state = ''.join(map(str, [0,0,1,1,1,1,1,1,0,0]))
		#print(self.q.get(state, 1), 0.0)
		#the state determined from camera 
		stateVal = getState(self, data)
		state = ''.join(map(str, stateVal))
		#for i in range(0, 2):
			#getQ(self, state, a)
		q = [getQ(self, state, a) for a in self.actions]
		maxQ = max(q)

		#If more than one Q
		count = q.count(maxQ)
		if count > 1:
			best = [i for i in range(len(self.actions)) if q[i] == maxQ]
			i = random.choice(best)
		else:
			i = q.index(maxQ)
		action = self.actions[i]

		#WRITE COMMAND
		vel_cmd = Twist()

		if action == 0:  # FORWARD
			vel_cmd.linear.x = 1
			vel_cmd.angular.z = 0.0
		elif action == 1:  # LEFT
			vel_cmd.linear.x = 0.0
			vel_cmd.angular.z = 0.3	
		elif action == 2:  # RIGHT
			vel_cmd.linear.x = 0.0
			vel_cmd.angular.z = -0.3
		print(vel_cmd)		
		self.vel_pub.publish(vel_cmd)		










rospy.init_node('control')
follower = Drive()
rospy.spin()








