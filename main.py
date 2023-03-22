import pandas as pd
import numpy as np
import quaternion

input_file = "Raw_data.csv"

# IMU Data

with open(input_file, 'r') as file:
    columns = ["IMU_Q0", "IMU_Q1", "IMU_Q2", "IMU_Q3"]
    IMU_df = pd.read_csv(file, header=0, usecols=columns)

# Specify first quaternion as the reference orientation from which to measure all rotations
IMU_ref = np.quaternion(IMU_df.values[0,0], IMU_df.values[0,1], IMU_df.values[0,2], IMU_df.values[0,3])
IMU_ref_conj = np.conjugate(IMU_ref)


# For every time sample, calculate the rotation quaterion, relative to the reference quat.
# N = len(IMU_df.index)
# a_1 = np.zeros((N, 4))
# a = as_quat_array(a)

# for row in IMU_df:
row = 0
IMU_i = np.quaternion(IMU_df.values[row,0], IMU_df.values[row,1], IMU_df.values[row,2], IMU_df.values[row,3])
IMU_rot = IMU_ref_conj*IMU_i
a = IMU_rot

row = 1
IMU_i_2 = np.quaternion(IMU_df.values[row,0], IMU_df.values[row,1], IMU_df.values[row,2], IMU_df.values[row,3])
IMU_rot_2 = IMU_ref_conj*IMU_i_2
b = IMU_rot_2

c = np.array([a,b])
d = quaternion.as_float(a)

print(c)
print(d)