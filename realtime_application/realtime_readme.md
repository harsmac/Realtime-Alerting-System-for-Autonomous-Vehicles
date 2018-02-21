# Readme for Realtime Alerting:

## Instructions to run the alerting system: 
1. Download the files from the link : https://drive.google.com/open?id=1TO6UEdBqJaRKV9AMWFd9nffi7oXk09Bk and store in current folder with the name as "final_dist".
2. The LBP features from the image subset of the Oxford Dataset have been extracted and put as pickle files : dusk_pickle, overcast_pickle, rain_pickle, sun_pickle. 
3. The file 'model_creation.py' trains a random forest on these pickle files and returns the model trained on it.It has already been created in this case and pickled as  : 'random_forest_model'.The script may be rerun to make changes.
4. The file 'realtime.py' runs the alerting system and it requires the presence of the final_dist folder with all the distances for VITTI dataset and the scores from the metric_codes folder.



