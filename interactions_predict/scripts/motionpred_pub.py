#!/usr/bin/env python

import sys
import rospy, math, time
import os, rospkg




import shutil # module to make operations on files
import time

#from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

#from sklearn import preprocessing
from std_msgs.msg import String,Int32,Int32MultiArray,MultiArrayLayout,MultiArrayDimension

from darko_prediction_msgs.msg import HumansPredictions, HumanMultiModalPredictions, HumanPrediction
from darko_interactions_msgs.msg import Interactions, scene_interactions
from darko_perception_msgs.msg import Humans, Human
from geometry_msgs.msg import PoseWithCovariance



#############################################################################################################################################
#                                                                                                                                           #
#############################################################################################################################################



class motion_pub:


	def __init__(self, ):

		rospy.init_node('darko_motion_pred', anonymous=True)
		self.rate = rospy.Rate(10)
		self.motionpub = rospy.Publisher('/hri/human_poses_prediction', HumansPredictions, queue_size=1000)
		self.perceptionpub = rospy.Publisher('/perception/humans', Humans, queue_size=1000)
		self.humans = 10
		self.dist_samples = 1
		self.time_horizon = 20



	def random_data(self, low=0, high = 0):

		return np.random.uniform(low, high)


	def motion_publisher(self,):

		rospy.loginfo("starting publishing humans poses per timestamp")


		human_multim_pred = HumanMultiModalPredictions()
		humans_pred = HumansPredictions()
		humans_pose = Humans()
		h_pose = PoseWithCovariance()



		for h in range(self.humans):
			print("h= ", h)
			for s in range(self.dist_samples):
				human_pred = HumanPrediction()
				human_pred.id = h
				human_pred.header.stamp = rospy.Time.now()

				for t in range(self.time_horizon):

					h_pose.pose.position.x = self.random_data(low = 0, high = 10)
					h_pose.pose.position.y = self.random_data(low = 0, high = 10)
					h_pose.pose.position.z = self.random_data(low = 0.8, high = 1.1)
					human_pred.sample.append(h_pose) # over trajectory for 1 human and 1 sample

					human_multim_pred.prediction.append(human_pred)

					human_pose = Human()
					human_pose.id = h
					human_pose.centroid.pose.position.x = self.random_data(low = 0, high = 10)
					human_pose.centroid.pose.position.y = self.random_data(low = 0, high = 10)
					human_pose.centroid.pose.position.z = self.random_data(low = 0.8, high = 1.1)
				
					humans_pose.humans.append(human_pose)


			humans_pred.predictions.append(human_multim_pred)
			print("human multi modal pred = ", len(humans_pred.predictions))

		now = rospy.get_rostime()
		rospy.loginfo("Current time %i %i", now.secs, now.nsecs)


		humans_pred.header.stamp = rospy.Time.now()

		humans_pose.header.stamp = rospy.Time.now()

		self.motionpub.publish(humans_pred)
		self.perceptionpub.publish(humans_pose)

         

if __name__ == '__main__':

	print("================================================== Get RANDOM INPUT FROM DARKO WP5 T5.1 ==================================================")
	input_interface = motion_pub() 

	while not rospy.is_shutdown():

		input_interface.motion_publisher()

		input_interface.rate.sleep()

	print("Finish publishing hand-coded motions")
