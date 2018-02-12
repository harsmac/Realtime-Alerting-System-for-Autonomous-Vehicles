# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:10:58 2017

@author: harshitha
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

X_dusk = pickle.load( open( "dusk_pickle", "rb" ) )
X_overcast = pickle.load( open( "overcast_pickle", "rb" ) )
X_rain = pickle.load( open( "rain_pickle", "rb" ) )
X_sun = pickle.load( open( "sun_pickle", "rb" ) )

X=X_dusk+X_overcast+X_rain+X_sun

Y_dusk=[]
Y_overcast=[]
Y_rain=[]
Y_sun=[]

for i in range(len(X_dusk)):
	Y_dusk+=[1]

for i in range(len(X_overcast)):
	Y_overcast+=[2]

for i in range(len(X_rain)):
	Y_rain+=[3]

for i in range(len(X_sun)):
	Y_sun+=[4]


Y=Y_dusk+Y_overcast+Y_rain+Y_sun

# LBP length is less than 26 so removing 
err_x=X[604]
err_y=Y[604]
X.remove(err_x)
Y.remove(err_y)

X_arr=np.array(X)
Y_arr=np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_arr, Y_arr, test_size=0.15, random_state=42)

# # Accuracy 90 with depth 5
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X_train, Y_train)
print "Accuracy on test data: ",clf.score(X_test,Y_test)

with open('random_forest_model', 'wb') as f:
	cPickle.dump(clf, f)
