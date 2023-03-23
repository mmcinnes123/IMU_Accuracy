import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

input_file = "Raw_data.csv"

# Define some quaternion functions
def quaternion_multiply(Q0, Q1):
    """
    Multiplies two quaternions.
    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31)
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32)
    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)
    """
    w0 = Q0[0]
    x0 = Q0[1]
    y0 = Q0[2]
    z0 = Q0[3]
    w1 = Q1[0]
    x1 = Q1[1]
    y1 = Q1[2]
    z1 = Q1[3]
    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])
    return final_quaternion

def quaternion_conjugate(Q0):
    w0 = Q0[0]
    x0 = Q0[1]
    y0 = Q0[2]
    z0 = Q0[3]
    output_quaternion = np.array([w0, -x0, -y0, -z0])
    return output_quaternion


# Create two data frames from data file - IMU and OptiTrack
with open(input_file, 'r') as file:
    columns_IMU = ["IMU_Q0", "IMU_Q1", "IMU_Q2", "IMU_Q3"]
    IMU_df = pd.read_csv(file, header=0, usecols=columns_IMU)

with open(input_file, 'r') as file:
    columns_Opt = ["Opt_Q0", "Opt_Q1", "Opt_Q2", "Opt_Q3"]
    Opt_df = pd.read_csv(file, header=0, usecols=columns_Opt)


# Specify first quaternion as the reference orientation from which to measure all rotations
IMU_ref = np.array((IMU_df.values[0,:4]))
IMU_ref_conj = quaternion_conjugate(IMU_ref)

Opt_ref = np.array((Opt_df.values[0,:4]))
Opt_ref_conj = quaternion_conjugate(Opt_ref)


# For every time sample, calculate the rotation quaterion, relative to the reference orientation (quat from first time sample).

N = len(IMU_df)
IMU_rotations = np.zeros((N,4))
Opt_rotations = np.zeros((N,4))

for row in range(len(IMU_df)):
    IMU_i = np.array((IMU_df.values[row,:4]))
    IMU_rotations[row] = quaternion_multiply(IMU_ref_conj,IMU_i)

for row in range(len(Opt_df)):
    Opt_i = np.array((Opt_df.values[row,:4]))
    Opt_rotations[row] = quaternion_multiply(Opt_ref_conj,Opt_i)


# For every rotation quaternion pair (IMU and OptiTrack), calculate the difference in orientation between these two quaternions
IMU_2_Opt_rot = np.zeros((N,4))

for i in range(len(Opt_rotations)):
    IMU_2_Opt_rot[i] = quaternion_multiply(quaternion_conjugate(Opt_rotations[i]), IMU_rotations[i])


# Save output data
np.savetxt('Output_data.csv', IMU_2_Opt_rot, delimiter=',', fmt='%.6f')
