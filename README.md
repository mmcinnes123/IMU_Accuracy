# IMU_Accuracy

A series of scripts developed during test of Delsys IMU accuracy relative to OptiTrack cluster data.

ReadQuatsOutputEuls takes raw quaternion data and allows the plotting of Euler angles based on a chosen decomposition sequence. 

main.py uses Raw_data.csv and uses quaternion multiplication to calculate a relative change in orientation by calculating rotation quaternion, in reference to the first quatenrion orientation of the data.

z_transformation.py just allows the application of some sort of transformation quaternion to all time samples, and outputs a csv file. 
