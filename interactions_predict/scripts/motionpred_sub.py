#!/usr/bin/env python


import sys
import rospy, math, time
import os, rospkg



import shutil # module to make operations on files
import time

#from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

#from Ind_qtc import QTC 
from qtc import qtcc22 
#from sklearn import preprocessing
from std_msgs.msg import String,Int32,Int32MultiArray,MultiArrayLayout,MultiArrayDimension

from darko_prediction_msgs.msg import HumansPredictions, HumanMultiModalPredictions, HumanPrediction
from darko_interactions_msgs.msg import Interactions, scene_interactions
from darko_perception_msgs.msg import Humans, Human
import message_filters
from geometry_msgs.msg import PoseWithCovariance


#############################################################################################################################################
#                                                                                                                                           #
#############################################################################################################################################


class qtc_pub:


	def __init__(self, ):

		rospy.init_node('darko_qtc_pred', anonymous=True)
		self.rate = rospy.Rate(50)
		self.humans = 0
		self.dist_samples = 1
		self.time_horizon = 20

		self.pubQTC = rospy.Publisher('/hri/interactions_prediction', scene_interactions, queue_size=10)
	
		self.subMotion = message_filters.Subscriber('/hri/human_poses_prediction', HumansPredictions)
		self.subperception = message_filters.Subscriber('/perception/humans', Humans)
		self.sync_sub = message_filters.ApproximateTimeSynchronizer([self.subMotion, self.subperception], queue_size =1000, slop=0.2)
		self.sync_sub.registerCallback(self.subsync_callback)
		self.time_horizon_motion = []
		self.current_motion = []
		self.past_history = 5
		self.past_data_header = -1
		self.past_perception_data_header = -1
		self.time_step = 0




	def subsync_callback(self, data_mot, data_percep):
			rospy.loginfo("starting to get current human motions")
			self.time_horizon = len(data_mot.predictions[0].prediction[0].sample)


			#while (len(self.current_motion) < self.time_horizon):
			if (data_percep.header.stamp != self.past_perception_data_header):
				self.current_motion.append(data_percep)
				self.past_perception_data_header = data_percep.header.stamp
				print("motion ahead len=", len(self.current_motion))



		#def scene_callback(self,data):
			rospy.loginfo("starting to accumulate time horizon motion predictions of multi-agents")

			#while (len(self.time_horizon_motion) != self.time_horizon):
			if (data_mot.header.stamp != self.past_data_header):
				self.time_horizon_motion.append(data_mot)
				self.past_data_header = data_mot.header.stamp
			


			print("first current motion we get is = ", self.current_motion[0].humans[0])
			self.humans = len(data_mot.predictions)
			qtc_pred = []


			rospy.loginfo("starting to formulate qtc")
			labels_qtc_pred = scene_interactions()

			for t in range(self.time_horizon):
				for i in range(self.humans -1):
					for j in range(i+1, self.humans):
						for s in range(self.dist_samples):

							label_qtc_pred = Interactions()
							qtc_pred_per_label = Int32MultiArray()

							ped_k_x_pred = self.time_horizon_motion[self.time_step].predictions[i].prediction[s].sample[t].pose.position.x
							ped_k_y_pred = self.time_horizon_motion[self.time_step].predictions[i].prediction[s].sample[t].pose.position.y
							ped_l_x_pred = self.time_horizon_motion[self.time_step].predictions[j].prediction[s].sample[t].pose.position.x
							ped_l_y_pred = self.time_horizon_motion[self.time_step].predictions[j].prediction[s].sample[t].pose.position.y
							#print("humans[i]=",self.current_motion[0].humans)
							ped_k_x_prev = self.current_motion[0].humans[i].centroid.pose.position.x
							ped_k_y_prev = self.current_motion[0].humans[i].centroid.pose.position.y
							ped_l_x_prev = self.current_motion[0].humans[j].centroid.pose.position.x
							ped_l_y_prev = self.current_motion[0].humans[j].centroid.pose.position.y
							qtc_kl_pred = qtcc22(np.array([[ped_k_x_prev, ped_k_y_prev],[ped_k_x_pred, ped_k_y_pred]]), np.array([[ped_l_x_prev, ped_l_y_prev],[ped_l_x_pred, ped_l_y_pred]]), False, False)
							print("qtc=", qtc_kl_pred.tolist()[0])
							qtc_pred.append(qtc_kl_pred.tolist()[0])


							qtc_pred_per_label.data  = qtc_kl_pred.tolist()[0]
				
							
							label_qtc_pred.label  = "label: ped1="+str(i)+"/ped2="+str(j)+str(self.time_step)+"/pred_step="+str(t+1)
							rospy.loginfo("label: ped1="+str(i)+"/ped2="+str(j)+"/tstep_now="+str(self.time_step)+"/pred_step="+str(t+1))

							label_qtc_pred.header.stamp = rospy.Time.now()
							
							label_qtc_pred.labels_qtc.append(qtc_pred_per_label)

							labels_qtc_pred.labels_qtc_series.append(label_qtc_pred)


				self.current_motion[0].humans[i].centroid.pose.position.x = self.time_horizon_motion[self.time_step].predictions[i].prediction[s].sample[t].pose.position.x
				self.current_motion[0].humans[i].centroid.pose.position.y = self.time_horizon_motion[self.time_step].predictions[i].prediction[s].sample[t].pose.position.y

				self.current_motion[0].humans[j].centroid.pose.position.x = self.time_horizon_motion[self.time_step].predictions[j].prediction[s].sample[t].pose.position.x
				self.current_motion[0].humans[j].centroid.pose.position.y = self.time_horizon_motion[self.time_step].predictions[j].prediction[s].sample[t].pose.position.y

			self.time_step +=1
			self.pubQTC.publish(labels_qtc_pred)     




if __name__ == '__main__':

    print("==================================================Get INPUT FROM DARKO WP5 T5.1 ==================================================")
    out_interface = qtc_pub() 

    while not rospy.is_shutdown():

        out_interface.rate.sleep()

    print("Finish importing scene data")
