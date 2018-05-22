# Results for few of the Oxford Dataset Images Explained: 

## Dusk Image-1:
![alt text](https://github.com/code-Assasin/Realtime-Alerting-System-for-Autonomous-Vehicles/blob/master/realtime_result_images/dusk1424450487165918_works.png)
###### Achieves a high score of 24 
This is because one car goes completely undetected. Upon closer inspection we find that, the centre of bounding box
is very far off from the actual center, hence resulting in a very high score.

## Dusk Image-2:
![alt text](https://github.com/code-Assasin/Realtime-Alerting-System-for-Autonomous-Vehicles/blob/master/realtime_result_images/dusk_1424450478229644.png)
###### Achieves a low score of 1 
In the case of the dusk image, clearly, the size of the bounding box is not correct, but its center seems to be near accurate and thus, justifying its score.

## Overcast Image-1:
![alt text](https://github.com/code-Assasin/Realtime-Alerting-System-for-Autonomous-Vehicles/blob/master/realtime_result_images/overcast_1403772884780659_works.png)
###### Achieves a low score of 3 
In this case clearly, the size of the bounding boxes is not correct, but the centers seems to be near accurate and thus, justifying its score.

## Overcast Image-2:
![alt text](https://github.com/code-Assasin/Realtime-Alerting-System-for-Autonomous-Vehicles/blob/master/realtime_result_images/overcast_1403772872155461.png)
###### Achieves a high score of 13
In this case clearly, the detector has clearly missed one of the cars and has identified the second one poorly and hence its much higher score.





