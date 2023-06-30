This series of scripts allows the reading of a Motion Monitor report file (quaternions) of one or more IMUs and an OMC cluster. 

The IMU and cluster quaternions can be transformed/adjusted in various ways in order to directly compare quaternion orientations at each time sample. 

Various outputs can be used to analyse results - Euler angles, RMSE based on quaternions, vector angle projections etc.

main.py can combined multiple repetitions to find average RMSE results. 
plot_quats.py just plots the quaternions and creates an APDM (can useful first before running full analysis to check timings and choose start and end time)

quat_functions.py defines quaternion functions such as quaternoin multiplication/conjugate/averages
analysis.py defines functions used to analyse results
pre_process.py defines functions used to transform IMU or cluster data
vel_vec_transform.py defines functions used to align global coordinate frames based on angular velocity vectors

function_test.py tests the function used to solve series of non-linear equations to find quaternion rotation from one vector to another

Unfinished.