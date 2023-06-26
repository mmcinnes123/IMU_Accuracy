This combination of scripts can be used to process Motion Monitor report files, for a single IMU with associated OMC cluster. 
Transform_Data_static.py applies a transformation to the IMU quaternions so that they are in the same GCF as the OMC data. 
(note, this is dependant on the specific settings in Motion Monitor). 
Vector_angle_static.py calculates the change in angle of a chosen axis of the LCFs relative to a chosen global axis. 
Comibine_static.py combines the final results. 
Vector_angle_static_plotting_all.py allows the angles to be plotted in a graph over time for visualisation.

Note: this script is designed to be run multiple times with file split into chunks (i.e. stationary periods).