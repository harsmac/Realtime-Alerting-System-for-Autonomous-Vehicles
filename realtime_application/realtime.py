# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:10:58 2017

@author: harshitha
Dscription : Program performs the realtime application for the AGV project.The code firstly loads the npy files of the VKITTI 
		metric scores and the final_dist folders npy files available at (https://drive.google.com/drive/folders/1TO6UEdBqJaRKV9AMWFd9nffi7oXk09Bk?usp=sharing).
	    The program then does a correlation based image search to find the closest image in the VKITTI dataset and finds 
	    the corresponding score.
"""
import cv2
import os
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
import pickle
import cPickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import csv
import sys
import requests
import skimage.io
import os
import glob
import pickle
import time

from IPython.display import display, Image, HTML
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage
from keras.models import Model
from keras import layers
import numpy as np
import pandas as pd
import scipy.sparse as sp
import skimage.io
import scipy
from keras import backend as K
import tensorflow as tf


start_time=time.time()


# LOAD IMAGE SCORES

# For 0001
dusk_score_0001 = np.load('/metric_codes/0001_scores/0001_sunset_scores.npy')
dusk_files_0001 = np.load('/metric_codes/0001_scores/0001_sunset_file_names.npy')

overcast_score_0001 = np.load('/metric_codes/0001_scores/0001_overcast_scores.npy')
overcast_files_0001 = np.load('/metric_codes/0001_scores/0001_overcast_file_names.npy')

rain_score_0001 = np.load('/metric_codes/0001_scores/0001_rain_scores.npy')
rain_files_0001 = np.load('/metric_codes/0001_scores/0001_rain_file_names.npy')

sun_score_0001 = np.load('/metric_codes/0001_scores/0001_morning_scores.npy')
sun_files_0001 = np.load('/metric_codes/0001_scores/0001_morning_file_names.npy')

# For 0002
dusk_score_0002 = np.load('/metric_codes/0002_scores/0002_sunset_scores.npy')
dusk_files_0002 = np.load('/metric_codes/0002_scores/0002_sunset_file_names.npy')

overcast_score_0002 = np.load('/metric_codes/0002_scores/0002_overcast_scores.npy')
overcast_files_0002 = np.load('/metric_codes/0002_scores/0002_overcast_file_names.npy')

rain_score_0002 = np.load('/metric_codes/0002_scores/0002_rain_scores.npy')
rain_files_0002 = np.load('/metric_codes/0002_scores/0002_rain_file_names.npy')

sun_score_0002 = np.load('/metric_codes/0002_scores/0002_morning_scores.npy')
sun_files_0002 = np.load('/metric_codes/0002_scores/0002_morning_file_names.npy')

# For 0006
dusk_score_0006 = np.load('/metric_codes/0006_scores/0006_sunset_scores.npy')
dusk_files_0006 = np.load('/metric_codes/0006_scores/0006_sunset_file_names.npy')

overcast_score_0006 = np.load('/metric_codes/0006_scores/0006_overcast_scores.npy')
overcast_files_0006 = np.load('/metric_codes/0006_scores/0006_overcast_file_names.npy')

rain_score_0006 = np.load('/metric_codes/0006_scores/0006_rain_scores.npy')
rain_files_0006 = np.load('/metric_codes/0006_scores/0006_rain_file_names.npy')

sun_score_0006 = np.load('/metric_codes/0006_scores/0006_morning_scores.npy')
sun_files_0006 = np.load('/metric_codes/0006_scores/0006_morning_file_names.npy')

# For 0018 
dusk_score_0018 = np.load('/metric_codes/0018_scores/0018_sunset_scores.npy')
dusk_files_0018 = np.load('/metric_codes/0018_scores/0018_sunset_file_names.npy')

overcast_score_0018 = np.load('/metric_codes/0018_scores/0018_overcast_scores.npy')
overcast_files_0018 = np.load('/metric_codes/0018_scores/0018_overcast_file_names.npy')

rain_score_0018 = np.load('/metric_codes/0018_scores/0018_rain_scores.npy')
rain_files_0018 = np.load('/metric_codes/0018_scores/0018_rain_file_names.npy')

sun_score_0018 = np.load('/metric_codes/0018_scores/0018_morning_scores.npy')
sun_files_0018 = np.load('/metric_codes/0018_scores/0018_morning_file_names.npy')

# For 0020
dusk_score_0020 = np.load('/metric_codes/0020_scores/0020_sunset_scores.npy')
dusk_files_0020 = np.load('/metric_codes/0020_scores/0020_sunset_file_names.npy')

overcast_score_0020 = np.load('/metric_codes/0020_scores/0020_overcast_scores.npy')
overcast_files_0020 = np.load('/metric_codes/0020_scores/0020_overcast_file_names.npy')

rain_score_0020 = np.load('/metric_codes/0020_scores/0020_rain_scores.npy')
rain_files_0020 = np.load('/metric_codes/0020_scores/0020_rain_file_names.npy')

sun_score_0020 = np.load('/metric_codes/0020_scores/0020_morning_scores.npy')
sun_files_0020 = np.load('/metric_codes/0020_scores/0020_morning_file_names.npy')

#####################################################################################################################################

str_index = ['0001','0002','0006','0018','0020']

# Function to get the nearest image's score from vkitti dataset
def score_pred(pred,folder_ind,file_ind):
	#print "FILE INDEX: ",file_ind
	# Folder 0001 
	if folder_ind == 0:
		print "0001 FOLDER"
		# dusk
		if pred == 1:
			print "File name: ",dusk_files_0001[file_ind]
			print "Corresponding Score: ",dusk_score_0001[file_ind]
			return dusk_score_0001[file_ind],dusk_files_0001[file_ind]
		# overcast
		elif pred == 2: 
			print "File name: ",overcast_files_0001[file_ind]
			print "Corresponding Score: ",overcast_score_0001[file_ind]
			return overcast_score_0001[file_ind],overcast_files_0001[file_ind]
		# rain
		elif pred == 3:
			print "File name: ",rain_files_0001[file_ind]
			print "Corresponding Score: ",rain_score_0001[file_ind]
			return rain_score_0001[file_ind],rain_files_0001[file_ind]
		else:
			print "File name: ",sun_files_0001[file_ind]
			print "Corresponding Score: ",sun_score_0001[file_ind]
			return sun_score_0001[file_ind],sun_files_0001[file_ind]

	# Folder 0002
	elif folder_ind == 1:
		print "0002 FOLDER"
		# dusk
		if pred == 1:
			print "File name: ",dusk_files_0002[file_ind]
			print "Corresponding Score: ",dusk_score_0002[file_ind]
			return dusk_score_0002[file_ind],dusk_files_0002[file_ind]
		# overcast
		elif pred == 2: 
			print "File name: ",overcast_files_0002[file_ind]
			print "Corresponding Score: ",overcast_score_0002[file_ind]
			return overcast_score_0002[file_ind],overcast_files_0002[file_ind]
		# rain
		elif pred == 3:
			print "File name: ",rain_files_0002[file_ind]
			print "Corresponding Score: ",rain_score_0002[file_ind]
			return rain_score_0002[file_ind],rain_files_0002[file_ind]
		else:
			print "File name: ",sun_files_0002[file_ind]
			print "Corresponding Score: ",sun_score_0002[file_ind]
			return sun_score_0002[file_ind],sun_files_0002[file_ind]

	# Folder 0006
	elif folder_ind == 2:
		print "0006 FOLDER"
		# dusk
		if pred == 1:
			print "File name: ",dusk_files_0006[file_ind]
			print "Corresponding Score: ",dusk_score_0006[file_ind]
			return dusk_score_0006[file_ind],dusk_files_0006[file_ind]
		# overcast
		elif pred == 2: 
			print "File name: ",overcast_files_0006[file_ind]
			print "Corresponding Score: ",overcast_score_0006[file_ind]
			return overcast_score_0006[file_ind],overcast_files_0006[file_ind]
		# rain
		elif pred == 3:
			print "File name: ",rain_files_0006[file_ind]
			print "Corresponding Score: ",rain_score_0006[file_ind]
			return rain_score_0006[file_ind],rain_files_0006[file_ind]
		else:
			print "File name: ",sun_files_0006[file_ind]
			print "Corresponding Score: ",sun_score_0006[file_ind]
			return sun_score_0006[file_ind],sun_files_0006[file_ind]

	# Folder 0018
	elif folder_ind == 3:
		print "0018 FOLDER"
		# dusk
		if pred == 1:
			print "File name: ",dusk_files_0018[file_ind]
			print "Corresponding Score: ",dusk_score_0018[file_ind]
			return dusk_score_0018[file_ind],dusk_files_0018[file_ind]
		# overcast
		elif pred == 2: 
			print "File name: ",overcast_files_0018[file_ind]
			print "Corresponding Score: ",overcast_score_0018[file_ind]
			return overcast_score_0018[file_ind],overcast_files_0018[file_ind]
		# rain
		elif pred == 3:
			print "File name: ",rain_files_0018[file_ind]
			print "Corresponding Score: ",rain_score_0018[file_ind]
			return rain_score_0018[file_ind],rain_files_0018[file_ind]
		else:
			print "File name: ",sun_files_0018[file_ind]
			print "Corresponding Score: ",sun_score_0018[file_ind]
			return sun_score_0018[file_ind],sun_files_0018[file_ind]

	# Folder 0020
	else:
		print "0020 FOLDER"
		# dusk
		if pred == 1:
			print "File name: ",dusk_files_0020[file_ind]
			print "Corresponding Score: ",dusk_score_0020[file_ind]
			return dusk_score_0020[file_ind],dusk_files_0020[file_ind]
		# overcast
		elif pred == 2: 
			print "File name: ",overcast_files_0020[file_ind]
			print "Corresponding Score: ",overcast_score_0020[file_ind]
			return overcast_score_0020[file_ind],overcast_files_0020[file_ind]
		# rain
		elif pred == 3:
			print "File name: ",rain_files_0020[file_ind]
			print "Corresponding Score: ",rain_score_0020[file_ind]
			return rain_score_0020[file_ind],rain_files_0020[file_ind]
		else:
			print "File name: ",sun_files_0020[file_ind]
			print "Corresponding Score: ",sun_score_0020[file_ind]
			return sun_score_0020[file_ind],sun_files_0020[file_ind]

# Finding closest/nearest image from vkitti dataset using correlation
def closest_vector(x_vector, vkitti_vectors):
	dist=[]
	for i in range(vkitti_vectors.shape[0]):
		dist+=[1-scipy.spatial.distance.correlation(x_vector,vkitti_vectors[i,:])]
	return np.argmin(dist),min(dist)

#################################################################################################################3

# LBP parameters
radius = 3
# Number of points to be considered as neighbours 
no_points = 8 * radius

# VGG 16 model till conv5_1
base_model = VGG16(include_top=False, weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block5_conv1').output)

# Unpickle the random forest model as rf
with open('random_forest_model', 'rb') as f:
	rf = cPickle.load(f)

# Load the vkitti images feature vectors 
# DUSK == SUNSET(1)
dusk_0001 = np.load('/realtime_application/final_dist/0001_dist/0001_dusk.npy')
dusk_0002 = np.load('/realtime_application/final_dist/0002_dist/0002_dusk.npy')
dusk_0006 = np.load('/realtime_application/final_dist/0006_dist/0006_dusk.npy')
dusk_0018 = np.load('/realtime_application/final_dist/0018_dist/0018_dusk.npy')
dusk_0020 = np.load('/realtime_application/final_dist/0020_dist/0020_dusk.npy')


# OVERCAST == CLOUDY (2)
overcast_0001 = np.load('/realtime_application/final_dist/0001_dist/0001_overcast.npy')
overcast_0002 = np.load('/realtime_application/final_dist/0002_dist/0002_overcast.npy')
overcast_0006 = np.load('/realtime_application/final_dist/0006_dist/0006_overcast.npy')
overcast_0018 = np.load('/realtime_application/final_dist/0018_dist/0018_overcast.npy')
overcast_0020 = np.load('/realtime_application/final_dist/0020_dist/0020_overcast.npy')

# RAIN (3)
rain_0001 = np.load('/realtime_application/final_dist/0001_dist/0001_rain.npy')
rain_0002 = np.load('/realtime_application/final_dist/0002_dist/0002_rain.npy')
rain_0006 = np.load('/realtime_application/final_dist/0006_dist/0006_rain.npy')
rain_0018 = np.load('/realtime_application/final_dist/0018_dist/0018_rain.npy')
rain_0020 = np.load('/realtime_application/final_dist/0020_dist/0020_rain.npy')

# SUN  == MORNING (4)
sun_0001 = np.load('/realtime_application/final_dist/0001_dist/0001_sun.npy')
sun_0002 = np.load('/realtime_application/final_dist/0002_dist/0002_sun.npy')
sun_0006 = np.load('/realtime_application/final_dist/0006_dist/0006_sun.npy')
sun_0018 = np.load('/realtime_application/final_dist/0018_dist/0018_sun.npy')
sun_0020 = np.load('/realtime_application/final_dist/0020_dist/0020_sun.npy')

################################################################################################################

X_test=[]
X_vectors = []

# Test folder path
folder_path='/realtime_application/test_images/tests/'

img_list=os.listdir(folder_path)
for fil in img_list[:]: # filelist[:] makes a copy of filelist.
	if not(fil.endswith(".png")):
		img_list.remove(fil)
		
print "List of images: ",img_list
print "LBP and VGG Features extraction.. "
i=0
while i<len(img_list):
	file_path=folder_path+img_list[i]
	im = cv2.imread(file_path)
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	
	# Extract LBP
	lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
	x = itemfreq(lbp.ravel())
	hist = x[:, 1]/sum(x[:, 1])
	X_test+=[hist]

	# Get vgg features 
	img = kimage.load_img(file_path,target_size=(224, 224))
	x1 = kimage.img_to_array(img)
	x1 = np.expand_dims(x1, axis=0)
	x1 = preprocess_input(x1)
	pred1 = model.predict(x1)
	x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None)(tf.convert_to_tensor(pred1))
	pred1_arr=tf.Session().run(x)
	pred1=pred1_arr.ravel()
	X_vectors+=[pred1]

	if i%10==0:
		print "iteration number: ",i

	i=i+1

print "Feature Extraction Completed\n"  


# Prediction using Random Forest 
X_test=np.array(X_test)
Y_pred = rf.predict(X_test)


for i in range(len(Y_pred)):
	print "For file: ",img_list[i]
	# If prediction is dusk
	if Y_pred[i]==1:
		print "Predicted DUSK"
		#Predict closest neighbour for each weather 		
		ind_0001,dist_0001 = closest_vector(X_vectors[i],dusk_0001)
		ind_0002,dist_0002 = closest_vector(X_vectors[i],dusk_0002)
		ind_0006,dist_0006 = closest_vector(X_vectors[i],dusk_0006)
		ind_0018,dist_0018 = closest_vector(X_vectors[i],dusk_0018)
		ind_0020,dist_0020 = closest_vector(X_vectors[i],dusk_0020)
		#Predict the closest among all the weather conditions 		
		dist_final = [dist_0001,dist_0002,dist_0006,dist_0018,dist_0020]
		ind_final = [ind_0001,ind_0002,ind_0006,ind_0018,ind_0020]
		#Get corresponding file name 		
		folder_index=np.argmin(dist_final)
		file_index=ind_final[folder_index]
		#Get corresponding score
		score_pred(1,folder_index,file_index)

	# Overcast
	elif Y_pred[i]==2:
		print "Predicted Overcast"
		ind_0001,dist_0001 = closest_vector(X_vectors[i],overcast_0001)
		ind_0002,dist_0002 = closest_vector(X_vectors[i],overcast_0002)
		ind_0006,dist_0006 = closest_vector(X_vectors[i],overcast_0006)
		ind_0018,dist_0018 = closest_vector(X_vectors[i],overcast_0018)
		ind_0020,dist_0020 = closest_vector(X_vectors[i],overcast_0020)

		dist_final = [dist_0001,dist_0002,dist_0006,dist_0018,dist_0020]
		ind_final = [ind_0001,ind_0002,ind_0006,ind_0018,ind_0020]

		folder_index=np.argmin(dist_final)
		file_index=ind_final[folder_index]

		score_pred(2,folder_index,file_index)

	# Rain
	elif Y_pred[i]==3:
		print "Predicted Rain"
		ind_0001,dist_0001 = closest_vector(X_vectors[i],rain_0001)
		ind_0002,dist_0002 = closest_vector(X_vectors[i],rain_0002)
		ind_0006,dist_0006 = closest_vector(X_vectors[i],rain_0006)
		ind_0018,dist_0018 = closest_vector(X_vectors[i],rain_0018)
		ind_0020,dist_0020 = closest_vector(X_vectors[i],rain_0020)

		dist_final = [dist_0001,dist_0002,dist_0006,dist_0018,dist_0020]
		ind_final = [ind_0001,ind_0002,ind_0006,ind_0018,ind_0020]

		folder_index=np.argmin(dist_final)
		file_index=ind_final[folder_index]

		score_pred(3,folder_index,file_index)

	# Sunny 
	else:
		print "Predcited Sun"
		ind_0001,dist_0001 = closest_vector(X_vectors[i],sun_0001)
		ind_0002,dist_0002 = closest_vector(X_vectors[i],sun_0002)
		ind_0006,dist_0006 = closest_vector(X_vectors[i],sun_0006)
		ind_0018,dist_0018 = closest_vector(X_vectors[i],sun_0018)
		ind_0020,dist_0020 = closest_vector(X_vectors[i],sun_0020)

		dist_final = [dist_0001,dist_0002,dist_0006,dist_0018,dist_0020]
		ind_final = [ind_0001,ind_0002,ind_0006,ind_0018,ind_0020]

		folder_index=np.argmin(dist_final)
		file_index=ind_final[folder_index]

		score_pred(4,folder_index,file_index)
	print "\n"

finish_time = time.time()
print "Time taken: ",finish_time-start_time," seconds"
