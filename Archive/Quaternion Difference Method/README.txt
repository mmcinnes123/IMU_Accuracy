This series of scripts allows the reading of Motion Monitor report file of one IMU relative to a OMC cluster. 
The quatenrion difference method can be used if a LCF for the cluster has been defined (with a stylus) which is aligned with the IMU casing. 
(And the global frames are also aligned). 

Transform.py applies the transformation to the IMU data so that it is expressed in same GCF as the OMC data. 
Main_analysis.py calcualtes the euler angles, the quaternion-difference based euler angles, and the smallest angle between two quaternions. 
The Combine.py files add up the results from each rep. 