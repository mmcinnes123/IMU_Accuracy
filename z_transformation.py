import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

input_file = "Raw_Data_from_Opti.csv"

# Define a quaterion multiplication function
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


# Define our multiplication quaternion
# -90 degree rotation around x-axis:
rot_quat = np.array([0.70711, -0.70711, 0, 0])

# Read in both the IMU and OptiTrack data (OptiTrack data will be unchanged and re-written into output file)
with open(input_file, 'r') as file:
    columns_IMU = ["IMU_Q0", "IMU_Q1", "IMU_Q2", "IMU_Q3"]
    IMU_df = pd.read_csv(file, header=10, usecols=columns_IMU)

with open(input_file, 'r') as file:
    columns_OpTr = ["OpTr_Q0", "OpTr_Q1", "OpTr_Q2", "OpTr_Q3"]
    OpTr_df = pd.read_csv(file, header=10, usecols=columns_OpTr)

# Apply the rotation by multipling by the quaternion rotation specified above
N = len(IMU_df)
transformed_quats = np.zeros((N, 4))

for row in range(len(IMU_df)):
    quat_i = np.array(IMU_df.values[row,:4])
    transformed_quats[row] = quaternion_multiply(rot_quat, quat_i)

# Turn OpTr data frame into a numpy array and combine OpTr new IMU data
OpTr_array = OpTr_df.to_numpy()
all_data = np.concatenate((transformed_quats, OpTr_array), axis=1)

# Output data to a csv file
np.savetxt('Transformed_data.csv', all_data, delimiter=',', fmt='%.6f', header="IMU_Q0,IMU_Q1,IMU_Q2,IMU_Q3,OpTr_Q0, OpTr_Q1, OpTr_Q2, OpTr_Q3", comments='')