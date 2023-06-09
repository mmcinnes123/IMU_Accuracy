#### This script reads in a raw txt file from a Motion Monitor report and applied some transformation to the IMU quaternions.
### This transformation is a transpose, then multiplication by a rotation matrix
### It outputs and new .txt file of the results and writes the data into a APDM style csv, ready for visulisation in OpenSim


from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd

### SETTINGS
# Make sure APDM file template is in same folder (AND FREQ = 100Hz!!)
input_file = "H_ST_R5 - Report1.txt"
tag = input_file.replace(" - Report1.txt", "")
output_file_name = tag + "_transformed.csv"
template_file = "APDM_template_3S.csv"

# Define a function for quaternion multiplication
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


### READ IN THE DATA
# Read in the IMU quaternions
with open(input_file, 'r') as file:
    columns_IMU = ["IMU_Q0", "IMU_Q1", "IMU_Q2", "IMU_Q3"]
    IMU_df = pd.read_csv(file, header=5, sep="\t", usecols=columns_IMU)

# Read in the OptiTrack cluster quaternions
with open(input_file, 'r') as file:
    columns_Clust = ["Clust_Q0", "Clust_Q1", "Clust_Q2", "Clust_Q3"]
    Clust_df = pd.read_csv(file, header=5, usecols=columns_Clust, sep="\t")
    Clust_df = Clust_df[:-1]

# Read in the stylus_defined cluster quaternions
with open(input_file, 'r') as file:
    columns_New_ax = ["New_ax_Q0", "New_ax_Q1", "New_ax_Q2", "New_ax_Q3"]
    New_ax_df = pd.read_csv(file, header=5, usecols=columns_New_ax, sep="\t")
    New_ax_df = New_ax_df[:-1]

# Read in the APDM template and save as an array
with open(template_file, 'r') as file:
    template_df = pd.read_csv(file, header=0)
    template_array = template_df.to_numpy()

### APPLY TRANSFORMATION TO IMU DATA

# Create the rotation matrix to transform the IMU orientations from Delsys global CF to OptiTrack global CF
rot_matrix = [[-1, 0, 0],[0, 0, 1],[0, 1, 0]]
# Turn the rotation matrix into a quaternion (note, scipy quats are scalar LAST)
rot_matrix_asR = R.from_matrix(rot_matrix)
rot_matrix_asquat = rot_matrix_asR.as_quat()
rot_quat = [rot_matrix_asquat[3], rot_matrix_asquat[0], rot_matrix_asquat[1], rot_matrix_asquat[2]]

# For every row in IMU data, take the transpose, then multiply by the rotation quaternion
N = len(IMU_df)-1
transformed_quats = np.zeros((N, 4))
for row in range(N):
    quat_i = np.array([IMU_df.values[row,0], -IMU_df.values[row,1], -IMU_df.values[row,2], -IMU_df.values[row,3]])
    transformed_quats[row] = quaternion_multiply(rot_quat, quat_i)


### WRITE NEW DATA TO APDM FILE TEMPLATE

# Make columns of zeros
zeros_25_df = pd.DataFrame(np.zeros((N, 25)))
zeros_11_df = pd.DataFrame(np.zeros((N, 11)))
zeros_2_df = pd.DataFrame(np.zeros((N, 2)))

# Make a dataframe with zeros columns inbetween the data
IMU_df_new = pd.DataFrame(transformed_quats)
IMU_and_zeros_df = pd.concat([zeros_25_df, IMU_df_new, zeros_11_df, Clust_df, zeros_11_df, New_ax_df, zeros_2_df], axis=1)

# Concatenate the IMU_and_zeros and the APDM template headings
IMU_and_zeros_array = IMU_and_zeros_df.to_numpy()
new_array = np.concatenate((template_array, IMU_and_zeros_array), axis=0)
new_df = pd.DataFrame(new_array)

# Add the new dataframe into the template
new_df.to_csv("Transformed_data_APDM_" + tag + ".csv", mode='w', index=False, header=False, encoding='utf-8')


### WRITE THE NEW IMU AND ORIGINAL CLUSTER DATA TO A .txt FILE

# Turn OpTr data frame into a numpy array and combine OpTr new IMU data
Clust_array = Clust_df.to_numpy()
New_ax_array = New_ax_df.to_numpy()
all_data = np.concatenate((transformed_quats, Clust_array, New_ax_array), axis=1)

# Output data to a csv file
np.savetxt(output_file_name, all_data, delimiter=',', fmt='%.6f', header="IMU_Q0,IMU_Q1,IMU_Q2,IMU_Q3,OpTr_Q0, OpTr_Q1, OpTr_Q2, OpTr_Q3, New_ax_Q0, New_ax_Q1, New_ax_Q2, New_ax_Q3", comments='')
