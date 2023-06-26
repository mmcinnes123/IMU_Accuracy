# This script contains all the functions needed to read in the raw data from the MotionMonitor report file, apply a transformation to the IMU data, then write the new results to a .txt file.
# This script also contains funcitons which writes the transformed data to an APDM template file, ready for viewing in OpenSim, and plots the quaternions in a graph.

from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Running " + "transform.py as " + __name__)


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


# Read in the IMU quaternions
def read_data_frame_from_file(input_file):
    with open(input_file, 'r') as file:
        df = pd.read_csv(file, header=5, sep="\t")
    # Make seperate data frames
    IMU1_df = df.filter(["IMU_Q0_T1", "IMU_Q1_T1", "IMU_Q2_T1", "IMU_Q3_T1"], axis=1)
    IMU2_df = df.filter(["IMU_Q0_T2", "IMU_Q1_T2", "IMU_Q2_T2", "IMU_Q3_T2"], axis=1)
    IMU3_df = df.filter(["IMU_Q0_T3", "IMU_Q1_T3", "IMU_Q2_T3", "IMU_Q3_T3"], axis=1)
    OpTr_Clus_df = df.filter(["Clust_Q0", "Clust_Q1", "Clust_Q2", "Clust_Q3"], axis=1)
    NewAx_Clus_df = df.filter(["New_ax_Q0", "New_ax_Q1", "New_ax_Q2", "New_ax_Q3"], axis=1)
    return IMU1_df, IMU2_df, IMU3_df, OpTr_Clus_df, NewAx_Clus_df


# Transform IMU data
def transform_IMU_data(IMU_df):
    # Create the rotation matrix to transform the IMU orientations from Delsys global CF to OptiTrack global CF
    rot_matrix = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    # Turn the rotation matrix into a quaternion (note, scipy quats are scalar LAST)
    rot_matrix_asR = R.from_matrix(rot_matrix)
    rot_matrix_asquat = rot_matrix_asR.as_quat()
    rot_quat = [rot_matrix_asquat[3], rot_matrix_asquat[0], rot_matrix_asquat[1], rot_matrix_asquat[2]]
    # For every row in IMU data, take the transpose, then multiply by the rotation quaternion
    N = len(IMU_df)
    transformed_quats = np.zeros((N, 4))
    for row in range(N):
        quat_i = np.array([IMU_df.values[row, 0], -IMU_df.values[row, 1], -IMU_df.values[row, 2], -IMU_df.values[row, 3]])
        transformed_quats[row] = quaternion_multiply(rot_quat, quat_i)
    transformed_quats_df = pd.DataFrame(transformed_quats)
    return transformed_quats_df


def interpolate_dfs(IMU1_df_trans, IMU2_df_trans, IMU3_df_trans, OpTr_Clus_df, NewAx_Clus_df):
    IMU1_df_trans = IMU1_df_trans.interpolate(limit=50)
    IMU2_df_trans = IMU2_df_trans.interpolate(limit=50)
    IMU3_df_trans = IMU3_df_trans.interpolate(limit=50)
    OpTr_Clus_df = OpTr_Clus_df.interpolate(limit=50)
    NewAx_Clus_df = NewAx_Clus_df.interpolate(limit=50)
    nan_count1 = IMU1_df_trans.isna().sum().sum()
    nan_count2 = IMU2_df_trans.isna().sum().sum()
    nan_count3 = IMU3_df_trans.isna().sum().sum()
    nan_count4 = OpTr_Clus_df.isna().sum().sum()
    nan_count5 = NewAx_Clus_df.isna().sum().sum()
    total_nans = nan_count5 + nan_count4 + nan_count3 + nan_count2 + nan_count1
    return IMU1_df_trans, IMU2_df_trans, IMU3_df_trans, OpTr_Clus_df, NewAx_Clus_df, total_nans


# Write new data to APDM file template
def write_to_APDM(IMU1_df_trans, IMU2_df_trans, IMU3_df_trans, Clust_df, template_file, tag):
    # Make columns of zeros
    N = len(IMU1_df_trans)
    zeros_25_df = pd.DataFrame(np.zeros((N, 25)))
    zeros_11_df = pd.DataFrame(np.zeros((N, 11)))
    zeros_2_df = pd.DataFrame(np.zeros((N, 2)))

    # Make a dataframe with zeros columns inbetween the data
    IMU_and_zeros_df = pd.concat([zeros_25_df, IMU1_df_trans, zeros_11_df, IMU2_df_trans, zeros_11_df, IMU3_df_trans, zeros_11_df, Clust_df, zeros_2_df], axis=1)

    # Read in the APDM template and save as an array
    with open(template_file, 'r') as file:
        template_df = pd.read_csv(file, header=0)
        template_array = template_df.to_numpy()

    # Concatenate the IMU_and_zeros and the APDM template headings
    IMU_and_zeros_array = IMU_and_zeros_df.to_numpy()
    new_array = np.concatenate((template_array, IMU_and_zeros_array), axis=0)
    new_df = pd.DataFrame(new_array)

    # Add the new dataframe into the template
    new_df.to_csv("Transformed_data_APDM_" + tag + ".csv", mode='w', index=False, header=False, encoding='utf-8', na_rep='nan')


# Write the transformed IMU data and original cluster data to a .txt file
def write_to_txt(IMU1_df_trans, IMU2_df_trans, IMU3_df_trans, OpTr_Clus_df, NewAx_Clus_df, tag):
    # Turn data frames into numpy arrays
    OpTr_Clus_array = OpTr_Clus_df.to_numpy()
    NewAx_Clus_array = NewAx_Clus_df.to_numpy()
    IMU1_array = IMU1_df_trans.to_numpy()
    IMU2_array = IMU2_df_trans.to_numpy()
    IMU3_array = IMU3_df_trans.to_numpy()
    all_data = np.concatenate((IMU1_array, IMU2_array, IMU3_array, OpTr_Clus_array, NewAx_Clus_array), axis=1)
    output_file_name = tag + "_transformed.csv"
    # Output data to a csv file
    np.savetxt(output_file_name, all_data, delimiter=',', fmt='%.6f',
               header="IMU1_Q0,IMU1_Q1,IMU1_Q2,IMU1_Q3,IMU2_Q0,IMU2_Q1,IMU2_Q2,IMU2_Q3,IMU3_Q0,IMU3_Q1,IMU3_Q2,IMU3_Q3,OpTr_Q0, OpTr_Q1, OpTr_Q2, OpTr_Q3, New_ax_Q0, New_ax_Q1, New_ax_Q2, New_ax_Q3",
               comments='')


# Plot the quaternions for comparison and checking timings
def plot_the_quats(IMU1_df_trans, NewAx_Clus_df, tag, sample_rate):
    # Turn data frames into numpy arrays
    NewAx_Clus_array = NewAx_Clus_df.to_numpy()
    IMU1_array = IMU1_df_trans.to_numpy()
    time = list(np.arange(0, len(NewAx_Clus_df) / sample_rate, 1 / sample_rate))
    plt.figure(1)
    fig = plt.figure(figsize=(10, 8))
    y1 = NewAx_Clus_array[:,0]
    y2 = NewAx_Clus_array[:,1]
    y3 = NewAx_Clus_array[:,2]
    y4 = NewAx_Clus_array[:,3]
    y5 = IMU1_array[:,0]
    y6 = IMU1_array[:,1]
    y7 = IMU1_array[:,2]
    y8 = IMU1_array[:,3]
    plt.scatter(time, y1, s=3, c='orange')
    plt.scatter(time, y2, s=3, c='fuchsia')
    plt.scatter(time, y3, s=3, c='red')
    plt.scatter(time, y4, s=3, c='maroon')
    plt.scatter(time, y5, s=3, c='blue')
    plt.scatter(time, y6, s=3, c='green')
    plt.scatter(time, y7, s=3, c='teal')
    plt.scatter(time, y8, s=3, c='darkgreen')
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    plt.title("Cluster and IMU2 Quaternions")
    plt.xlabel('Time')
    plt.ylabel('Quats')
    plt.grid(axis="x", which="both")
    x_range = round(len(time)/sample_rate)
    plt.xticks(range(0, x_range, 1))
    plt.legend(["Clus_Q0", "Clust_Q1", "Clust_Q2", "Clust_Q3", "IMU_Q0", "IMU_Q1", "IMU_Q2", "IMU_Q3"], loc="lower right")
    plt.savefig("RefQuats_" + tag + ".png")
    plt.clf()



# Unit test of code in this file
if __name__ == "__main__":
    # Make sure APDM file template is in same folder (AND FREQ = 100Hz!!)
    input_file = "Inter_IMU_R1 - Report1.txt"
    tag = input_file.replace(" - Report1.txt", "")
    template_file = "APDM_template_4S.csv"
    sample_rate = 100
    plot_quats = True
    quats_inverse = True
    IMU1_df, IMU2_df, IMU3_df, OpTr_Clus_df, NewAx_Clus_df = read_data_frame_from_file(input_file)
    IMU1_df_trans = transform_IMU_data(IMU1_df)
    IMU2_df_trans = transform_IMU_data(IMU2_df)
    IMU3_df_trans = transform_IMU_data(IMU3_df)
    write_to_APDM(IMU1_df_trans, IMU2_df_trans, IMU3_df_trans, NewAx_Clus_df, template_file, tag)
    write_to_txt(IMU1_df_trans, IMU2_df_trans, IMU3_df_trans, OpTr_Clus_df, NewAx_Clus_df, tag)
    plot_the_quats(IMU1_df_trans, NewAx_Clus_df, tag, sample_rate)
