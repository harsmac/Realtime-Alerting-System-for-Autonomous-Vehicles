# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:10:58 2017

@author: harshitha
Dscription : Program Scores the list of images from a folder wrt its clone.Change the file paths of the clone and bad weather 
		to make it run. The scores are finally saved as npy files.
"""

import numpy as np
import re

# Parameter definition
# Very Strict Penalty
Lambda=5000
# Liberal Penalty
# Lambda = 500
# Image length and breadth
L=375
B=1242
LB=float(L*B)

# Function for the metric
def metric(x1,x2,sigma_1,sigma_2):
	if sigma_1!=sigma_2:
		A=np.array([[(sigma_1+sigma_2)/2.0,min(sigma_1,sigma_2)],[min(sigma_1,sigma_2),max(sigma_1,sigma_2)]])
		t=np.dot(np.dot((x1-x2),A),(x1-x2).T)
		# print "t value.... ",t[0,0]
		return t[0,0]
# 	If the confidence values are equal
	else:
		t=np.dot((x1-x2),(x1-x2).T)
		# print t[0,0]
		return t[0,0]



# Parse the following files : Change the file paths
clone_file_path='/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/gpu_backup/predictions/0020_results/result_0020_clone.txt'
bad_weather_path='/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/gpu_backup/predictions/0020_results/result_0020_sunset.txt'


#  FOR CLONE ..

# Open the file with read only permit
f = open(clone_file_path, "r")
# use readlines to read all lines in the file
# The variable "lines" is a list containing all lines in the file
lines = f.readlines()
# close the file after reading the lines.
f.close()

lookup = 'Enter '
enter_lines=[]

# Collect lines where "Enter Image Path" has occured..
with open(clone_file_path) as myFile:
	for num, line in enumerate(myFile, 1):
		if lookup in line:
			enter_lines+=[num-1]

list_final_clone=[]
for i in range(len(enter_lines)-1):

	# Search in these lines enter_lines[i]+1 to enter_lines[i+1]-1

	temp_list=[]

	# Get file name
	file_name_str=re.findall(r'\d+',lines[enter_lines[i]])
	file_name_str=file_name_str[1]+'.png'
	file_name=[file_name_str]

	# Parse for confidence(sigma) values
	confidence_list=[]
	for j in range(enter_lines[i]+1,enter_lines[i+1]-1+1,2):
		confidence_list+=[int(lines[j][5:7])/100.0]

	# Parse for list of coordinates
	coords_list=[]
	for k in range(enter_lines[i]+2,enter_lines[i+1]-1+1,2):
		str_coords=lines[k][24:-1]
		# Remove spaces
		str_coords.replace(" ","")
		# Find all ints in string
		str_coords = re.findall(r'\d+',str_coords)
		int_coords = map(int,str_coords)
		coords_list+=[int_coords]

	temp_list=[file_name,confidence_list,coords_list]

	list_final_clone+=[temp_list]


# # FOR BAD WEATHER 

# Open the file with read only permit
f = open(bad_weather_path, "r")
# use readlines to read all lines in the file
# The variable "lines" is a list containing all lines in the file
lines = f.readlines()
# close the file after reading the lines.
f.close()

lookup = 'Enter '
enter_lines=[]

with open(bad_weather_path) as myFile:
    for num, line in enumerate(myFile, 1):
        if lookup in line:
            #print 'found at line:', num
            enter_lines+=[num-1]

list_final_bad=[]
for i in range(len(enter_lines)-1):

	# Search in these lines enter_lines[i]+1 to enter_lines[i+1]-1

	temp_list=[]
	# Get file name
	file_name_str=re.findall(r'\d+',lines[enter_lines[i]])
	file_name_str=file_name_str[1]+'.png'
	file_name=[file_name_str]

	# Parse for confidence(sigma) values
	confidence_list=[]
	for j in range(enter_lines[i]+1,enter_lines[i+1]-1+1,2):
		confidence_list+=[int(lines[j][5:7])/100.0]

	# Parse for coordinates list
	coords_list=[]
	for k in range(enter_lines[i]+2,enter_lines[i+1]-1+1,2):
		str_coords=lines[k][24:-1]
		# Remove spaces
		str_coords.replace(" ","")
		# Find all ints in string
		str_coords = re.findall(r'\d+',str_coords)
		int_coords = map(int,str_coords)
		coords_list+=[int_coords]

	temp_list=[file_name,confidence_list,coords_list]

	list_final_bad+=[temp_list]



Score_list=[]
file_name_list = []

# For all images in the folder 
for i in range(len(list_final_bad)):

	Score = 0
	# Score calculated for one image
	mini_good=list_final_clone[i]
	mini_bad=list_final_bad[i]
	file_name_list+=[mini_bad[0][0]]

	# print mini_good
	# print mini_bad

	# No object is detected at all..
	if len(mini_bad[1])==0:
		# print "Case 1"
		num_objects=len(mini_good[1])
		area=0
		for j in range(num_objects):
			# Get width and height
			wid = mini_good[2][j][2]
			hig = mini_good[2][j][3]
			area = area + wid * hig
		# All objects are missing .. hence score = Lambda*(w1h1+ w2h2 + w3h3 + ...)/LB
		Score = area*Lambda
		Score = Score/LB
		Score_list+=[Score]

	# Missing objects
	elif len(mini_good[1])>len(mini_bad[1]):
		# print "Case 2"
		# print mini_good

		# For each object detected in the bad image
		for k in range(len(mini_bad[1])):
			# print k
			x_coord=mini_bad[2][k][0]
			y_coord=mini_bad[2][k][1]	
			xy_bad=np.array([[x_coord,y_coord]])
			sigma_bad = mini_bad[1][k]
			min_dist=1000000
			min_index=-100
			# Find the corresponding object detected in the good image
			for l in range(len(mini_good[1])):
				x_coord=mini_good[2][l][0]
				y_coord=mini_good[2][l][1]	
				xy_good=np.array([[x_coord,y_coord]])
				dist = np.linalg.norm(xy_good-xy_bad)
				if(dist<min_dist):
					min_dist = dist
					min_index = l

			# For kth object in the bad image the closest is the min_index th one in the clone
			sigma_good= mini_good[1][min_index]
			wid = mini_good[2][min_index][2]
			hig = mini_good[2][min_index][3]
			x_coord=mini_good[2][min_index][0]
			y_coord=mini_good[2][min_index][1]	
			xy_good=np.array([[x_coord,y_coord]])
			Score = Score + wid*hig*metric(xy_bad,xy_good,sigma_bad,sigma_good)
			# To reduce computation expenses pop out already matched elements
			del mini_good[1][min_index]
			del mini_good[2][min_index]

		# Once complete for existing objects account for missing ones with lambda
		num_objects_remaining=len(mini_good[1])
		area=0
		for j in range(num_objects_remaining):
			wid = mini_good[2][j][2]
			hig = mini_good[2][j][3]
			area = area + wid * hig
		Score = Score + area*Lambda
		Score = Score/LB
		Score_list+=[Score]

	# Same number of objects are detected in both cases
	elif len(mini_good[1]) == len(mini_bad[1]) : 
		# print "Case 3"
		for k in range(len(mini_bad[1])):
			# print k
			x_coord=mini_bad[2][k][0]
			y_coord=mini_bad[2][k][1]	
			xy_bad=np.array([[x_coord,y_coord]])
			sigma_bad = mini_bad[1][k]
			min_dist=1000000
			min_index=-100
			# Find the corresponding nearest element in the good image
			for l in range(len(mini_good[1])):
				x_coord=mini_good[2][l][0]
				y_coord=mini_good[2][l][1]	
				xy_good=np.array([[x_coord,y_coord]])
				dist = np.linalg.norm(xy_good-xy_bad)
				if(dist<min_dist):
					min_dist = dist
					min_index = l

			# For kth object in the bad image the closest is the lth one in the clone
			sigma_good= mini_good[1][min_index]
			wid = mini_good[2][min_index][2]
			hig = mini_good[2][min_index][3]

			x_coord=mini_good[2][min_index][0]
			y_coord=mini_good[2][min_index][1]	
			xy_good=np.array([[x_coord,y_coord]])
			Score = Score + wid*hig*metric(xy_bad,xy_good,sigma_bad,sigma_good)
			# To reduce computation expenses pop out already matched elements
			del mini_good[1][min_index]
			del mini_good[2][min_index]

		Score = Score/LB
		Score_list+=[Score]

	# Very rare case if the image quality of bad condition is better than the clone 
	elif len(mini_good[1])<len(mini_bad[1]):
		# print "Case 4"

		# Now instead we match every good object to corresponding bad one to measure distortion from our ideal case..(clone)
		for k in range(len(mini_good[1])):
			# print k
			x_coord=mini_good[2][k][0]
			y_coord=mini_good[2][k][1]	
			xy_good=np.array([[x_coord,y_coord]])
			sigma_good = mini_good[1][k]
			min_dist=1000000
			min_index=-100
			# Find the corresponding object in the bad image
			for l in range(len(mini_bad[1])):
				x_coord=mini_bad[2][l][0]
				y_coord=mini_bad[2][l][1]	
				xy_bad=np.array([[x_coord,y_coord]])
				dist = np.linalg.norm(xy_good-xy_bad)
				if(dist<min_dist):
					min_dist = dist
					min_index = l

			# For kth object in the good image the closest is the lth one in the bad one
			sigma_bad= mini_bad[1][min_index]
			# The width and height are always wrt to the good image
			wid = mini_good[2][k][2]
			hig = mini_good[2][k][3]

			x_coord=mini_bad[2][min_index][0]
			y_coord=mini_bad[2][min_index][1]	
			xy_bad=np.array([[x_coord,y_coord]])
			Score = Score + wid*hig*metric(xy_bad,xy_good,sigma_bad,sigma_good)
			# To reduce computation expenses pop out already matched elements
			del mini_bad[1][min_index]
			del mini_bad[2][min_index]

		Score = Score/LB
		Score_list+=[Score]

	else: 
		print "None of the above cases.."

Score_array = np.array(Score_list)
np.save('0020_sunset_scores', Score_array)

file_name_arr = np.array(file_name_list)
np.save('0020_sunset_file_names',file_name_arr)
