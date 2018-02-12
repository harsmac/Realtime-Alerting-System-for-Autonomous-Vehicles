import numpy as np

# DUSK == SUNSET(1)
dusk_0001 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0001_dist/0001_sunset.npy')[0:425]
dusk_0002 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0002_dist/0002_sunset.npy')
dusk_0006 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0006_dist/0006_sunset.npy')[0:175]
dusk_0018 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0018_dist/0018_sunset.npy')
dusk_0020 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0020_dist/0020_sunset.npy')

# OVERCAST == CLOUDY (2)
overcast_0001 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0001_dist/0001_overcast.npy')[0:425]
overcast_0002 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0002_dist/0002_overcast.npy')
overcast_0006 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0006_dist/0006_overcast.npy')[0:175]
overcast_0018 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0018_dist/0018_overcast.npy')
overcast_0020 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0020_dist/0020_overcast.npy')

# RAIN (3)
rain_0001 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0001_dist/0001_rain.npy')[0:425]
rain_0002 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0002_dist/0002_rain.npy')
rain_0006 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0006_dist/0006_rain.npy')[0:175]
rain_0018 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0018_dist/0018_rain.npy')
rain_0020 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0020_dist/0020_rain.npy')

# SUN  == MORNING (4)
sun_0001 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0001_dist/0001_morning.npy')[0:425]
sun_0002 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0002_dist/0002_morning.npy')
sun_0006 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0006_dist/0006_morning.npy')[0:175]
sun_0018 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0018_dist/0018_morning.npy')
sun_0020 = np.load('/home/harshitha/Desktop/pls_2/AGV_project/ICIP_2018/realtime_application/actual_dist/0020_dist/0020_morning.npy')

# Preprocess 0002 folder
l=range(10,51+1)
dusk_0002=np.delete(dusk_0002,l,axis=0)
overcast_0002=np.delete(overcast_0002,l,axis=0)
rain_0002=np.delete(rain_0002,l,axis=0)
sun_0002=np.delete(sun_0002,l,axis=0)

l=range(163,185+1)
dusk_0002=np.delete(dusk_0002,l,axis=0)
overcast_0002=np.delete(overcast_0002,l,axis=0)
rain_0002=np.delete(rain_0002,l,axis=0)
sun_0002=np.delete(sun_0002,l,axis=0)

# Preprocess 0006 folder
l=range(30,37+1)
dusk_0006=np.delete(dusk_0006,l,axis=0)
overcast_0006=np.delete(overcast_0006,l,axis=0)
rain_0006=np.delete(rain_0006,l,axis=0)
sun_0006=np.delete(sun_0006,l,axis=0)

l=range(137,142+1)
dusk_0006=np.delete(dusk_0006,l,axis=0)
overcast_0006=np.delete(overcast_0006,l,axis=0)
rain_0006=np.delete(rain_0006,l,axis=0)
sun_0006=np.delete(sun_0006,l,axis=0)

# Preprocess 0018 folder
l=range(0,27+1)
dusk_0018=np.delete(dusk_0018,l,axis=0)
overcast_0018=np.delete(overcast_0018,l,axis=0)
rain_0018=np.delete(rain_0018,l,axis=0)
sun_0018=np.delete(sun_0018,l,axis=0)

l=range(41,58+1)
dusk_0018=np.delete(dusk_0018,l,axis=0)
overcast_0018=np.delete(overcast_0018,l,axis=0)
rain_0018=np.delete(rain_0018,l,axis=0)
sun_0018=np.delete(sun_0018,l,axis=0)

np.save('0001_dusk', dusk_0001)
np.save('0001_overcast', overcast_0001)
np.save('0001_rain', rain_0001)
np.save('0001_sun', sun_0001)

np.save('0002_dusk', dusk_0002)
np.save('0002_overcast', overcast_0002)
np.save('0002_rain', rain_0002)
np.save('0002_sun', sun_0002)

np.save('0006_dusk', dusk_0006)
np.save('0006_overcast', overcast_0006)
np.save('0006_rain', rain_0006)
np.save('0006_sun', sun_0006)

np.save('0018_dusk', dusk_0018)
np.save('0018_overcast', overcast_0018)
np.save('0018_rain', rain_0018)
np.save('0018_sun', sun_0018)

np.save('0020_dusk', dusk_0020)
np.save('0020_overcast', overcast_0020)
np.save('0020_rain', rain_0020)
np.save('0020_sun', sun_0020)

