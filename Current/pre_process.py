# This script contains all the functions needed to read in the raw data from the MotionMonitor report file, apply a transformation to the IMU data, then write the new results to a .txt file.
# This script also contains funcitons which writes the transformed data to an APDM template file, ready for viewing in OpenSim, and plots the quaternions in a graph.

from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import root
from quat_functions import quaternion_multiply
from quat_functions import quaternion_conjugate
from quat_functions import average_quaternions
from quat_functions import ang_vel_from_quats


print("Running " + "pre_process.py as " + __name__)


# Read all data in from specified input file
def read_data_frame_from_file(input_file):
    with open(input_file, 'r') as file:
        df = pd.read_csv(file, header=5, sep="\t")
    # Make seperate data frames
    IMU1_df = df.filter(["IMU_Q0", "IMU_Q1", "IMU_Q2", "IMU_Q3"], axis=1)
    OpTr_Clus_df = df.filter(["Clust_Q0", "Clust_Q1", "Clust_Q2", "Clust_Q3"], axis=1)
    NewAx_Clus_df = df.filter(["New_ax_Q0", "New_ax_Q1", "New_ax_Q2", "New_ax_Q3"], axis=1)
    return IMU1_df, OpTr_Clus_df, NewAx_Clus_df


# Transform IMU data into Y-up convention
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


# Interpolate all the data and return how many missing data points there were.
def interpolate_dfs(IMU1_df_trans, OpTr_Clus_df, NewAx_Clus_df):
    IMU1_df_trans = IMU1_df_trans.interpolate(limit=50)
    OpTr_Clus_df = OpTr_Clus_df.interpolate(limit=50)
    NewAx_Clus_df = NewAx_Clus_df.interpolate(limit=50)
    nan_count1 = IMU1_df_trans.isna().sum().sum()
    nan_count5 = OpTr_Clus_df.isna().sum().sum()
    nan_count6 = NewAx_Clus_df.isna().sum().sum()
    total_nans = nan_count5 + nan_count1 + nan_count6
    return IMU1_df_trans, OpTr_Clus_df, NewAx_Clus_df, total_nans


# Trim the data frames based on start and end time
def trim_df(df, start_time, end_time, sample_rate):
    first_index = int(start_time*sample_rate)
    last_index = int(end_time*sample_rate)
    index_range = list(range(first_index, last_index))
    df_new = df.iloc[index_range, :]
    df_new_new = df_new.reset_index(drop=True)
    return df_new_new


# Transform cluster quaternions based on a rotation quaternion between IMU and Cluster at t=0
def trans_clust_t0(Clus_df, IMU_df):
    q_clus_t0 = np.array([Clus_df.values[0, 0], Clus_df.values[0, 1], Clus_df.values[0, 2], Clus_df.values[0, 3]])
    q_IMU_t0 = np.array([IMU_df.values[0, 0], IMU_df.values[0, 1], IMU_df.values[0, 2], IMU_df.values[0, 3]])
    q_rot_clust2IMU = quaternion_multiply(quaternion_conjugate(q_clus_t0), q_IMU_t0)
    N = len(Clus_df)
    trans_Clus_quats = np.zeros((N, 4))
    for row in range(N):
        clust_quat_i = np.array([Clus_df.values[row, 0], Clus_df.values[row,1], Clus_df.values[row,2], Clus_df.values[row,3]])
        trans_Clus_quats[row] = quaternion_multiply(clust_quat_i, q_rot_clust2IMU)
    transformed_clust_quats = pd.DataFrame(trans_Clus_quats)
    return transformed_clust_quats


# Transform cluster quaternions based on an average rotation quaternion between IMU and Cluster
def trans_clust_average(Clus_df, IMU_df):
    N = len(Clus_df)
    q_rot_clust2IMU = np.zeros((N, 4))
    for row in range(N):
        q_clust = Clus_df.values[row, :]
        q_IMU = IMU_df.values[row, :]
        q_rot_clust2IMU[row] = quaternion_multiply(quaternion_conjugate(q_clust), q_IMU)
    q_rot_average = average_quaternions(q_rot_clust2IMU)
    trans_Clus_quats = np.zeros((N, 4))
    for row in range(N):
        clust_quat_i = np.array([Clus_df.values[row, 0], Clus_df.values[row,1], Clus_df.values[row,2], Clus_df.values[row,3]])
        trans_Clus_quats[row] = quaternion_multiply(clust_quat_i, q_rot_average)
    transformed_clust_quats = pd.DataFrame(trans_Clus_quats)
    return transformed_clust_quats


# Find the rotation quaternions from Cluster to IMU based on angular velocity vectors
def vel_rot_quat_GCF(IMU_df, Clust_df):

    N = len(IMU_df)
    vel_rot_quats = np.empty((0, 4))

    for i in range(N-1):
        # Specify quats 0 and 1 based on chosen time points
        time_step = i + 1
        clus_t0_quat = Clust_df.values[i]
        clus_t1_quat = Clust_df.values[time_step]
        IMU_t0_quat = IMU_df.values[i]
        IMU_t1_quat = IMU_df.values[time_step]

        # Calculate angular velocity in LCF based on quats 0 and 1
        wM = ang_vel_from_quats(clus_t0_quat, clus_t1_quat, 0.01)
        wI = ang_vel_from_quats(IMU_t0_quat, IMU_t1_quat, 0.01)

        # Transform into angular velocity in GCF
        wM_global_asquat = quaternion_multiply(quaternion_multiply(clus_t0_quat, np.concatenate(([0], wM))), quaternion_conjugate(clus_t0_quat))
        wM_global = np.delete(wM_global_asquat, 0)
        wI_global_asquat = quaternion_multiply(quaternion_multiply(IMU_t0_quat, np.concatenate(([0], wI))), quaternion_conjugate(IMU_t0_quat))
        wI_global = np.delete(wI_global_asquat, 0)

        # Apply a cutt off threshold based on magnitude of angular velocity
        # # if np.linalg.norm(clus_vel_col_b) >= 3 and np.linalg.norm(IMU_vel_col_b) >= 3:
        # rot_2_apply_mat = np.matmul(IMU_vel_col_b, np.linalg.pinv(clus_vel_col_b))
        # rot_2_apply_R = R.from_matrix(rot_2_apply_mat)
        # rot_2_apply_R_quat = rot_2_apply_R.as_quat()
        # vel_rot_quats = np.append(vel_rot_quats, np.array([[rot_2_apply_R_quat[3], rot_2_apply_R_quat[0], rot_2_apply_R_quat[1], rot_2_apply_R_quat[2]]]), axis=0)

        wM_x = wM_global[0]
        wM_y = wM_global[1]
        wM_z = wM_global[2]
        wI_x = wI_global[0]
        wI_y = wI_global[1]
        wI_z = wI_global[2]
        weight = 10.0

        # Define a function to solve the non-linear systems of equations
        def f(wxyz):
            q0 = wxyz[0]
            q1 = wxyz[1]
            q2 = wxyz[2]
            q3 = wxyz[3]

            f1 = (wM_x - wI_x) * q0 - (wM_z + wI_z) * q2 + (wM_y + wI_y) * q3
            f2 = (wM_y - wI_y) * q0 + (wM_z + wI_z) * q1 - (wM_x + wI_x) * q3
            f3 = (wM_z - wI_z) * q0 - (wM_y + wI_y) * q1 + (wM_x + wI_x) * q2
            f4 = weight * (q0**2 + q1**2 + q2**2 + q3**2 - 1)

            return np.array([f1, f2, f3, f4])

        wxyz_0 = np.array([0, 1.0, 0, 0])
        wxyz = fsolve(f, wxyz_0)
        wxyz_norm = wxyz / np.linalg.norm(wxyz)
        vel_rot_quats = np.append(vel_rot_quats, np.array([wxyz_norm]), axis=0)

        print(i)
        print(wM_global)
        print(wI_global)
        print(wxyz_norm)

    rot_2_apply = vel_rot_quats

    return rot_2_apply


# Find the rotation quaternions from Cluster to IMU based on angular velocity vectors
def vel_rot_quat_LCF(IMU_df, Clust_df):

    N = len(IMU_df)
    vel_rot_quats = np.empty((0, 4))

    for i in range(N-1):
        # Specify quats 0 and 1 based on chosen time points
        time_step = i + 1
        clus_t0_quat = Clust_df.values[i]
        clus_t1_quat = Clust_df.values[time_step]
        IMU_t0_quat = IMU_df.values[i]
        IMU_t1_quat = IMU_df.values[time_step]

        # Calculate angular velocity in LCF based on quats 0 and 1
        wM = ang_vel_from_quats(clus_t0_quat, clus_t1_quat, 0.01)
        wI = ang_vel_from_quats(IMU_t0_quat, IMU_t1_quat, 0.01)

        # Apply a cutt off threshold based on magnitude of angular velocity
        # # if np.linalg.norm(clus_vel_col_b) >= 3 and np.linalg.norm(IMU_vel_col_b) >= 3:
        # rot_2_apply_mat = np.matmul(IMU_vel_col_b, np.linalg.pinv(clus_vel_col_b))
        # rot_2_apply_R = R.from_matrix(rot_2_apply_mat)
        # rot_2_apply_R_quat = rot_2_apply_R.as_quat()
        # vel_rot_quats = np.append(vel_rot_quats, np.array([[rot_2_apply_R_quat[3], rot_2_apply_R_quat[0], rot_2_apply_R_quat[1], rot_2_apply_R_quat[2]]]), axis=0)

        wM_x = wM[0]
        wM_y = wM[1]
        wM_z = wM[2]
        wI_x = wI[0]
        wI_y = wI[1]
        wI_z = wI[2]
        weight = 10.0

        # Define a function to solve the non-linear systems of equations
        def f(wxyz):
            q0 = wxyz[0]
            q1 = wxyz[1]
            q2 = wxyz[2]
            q3 = wxyz[3]

            f1 = (wM_x - wI_x) * q0 - (wM_z + wI_z) * q2 + (wM_y + wI_y) * q3
            f2 = (wM_y - wI_y) * q0 + (wM_z + wI_z) * q1 - (wM_x + wI_x) * q3
            f3 = (wM_z - wI_z) * q0 - (wM_y + wI_y) * q1 + (wM_x + wI_x) * q2
            f4 = weight * (q0**2 + q1**2 + q2**2 + q3**2 - 1)

            return np.array([f1, f2, f3, f4])

        wxyz_0 = np.array([0, 1.0, 0, 0])
        wxyz = root(f, wxyz_0, method='lm')

        if wxyz.success == True:
            wxyz_norm = wxyz.x / np.linalg.norm(wxyz.x)
            vel_rot_quats = np.append(vel_rot_quats, np.array([wxyz_norm]), axis=0)

        print(i)
        print(wM)
        print(wI)
        print(wxyz_norm)

    rot_2_apply = vel_rot_quats

    return rot_2_apply


# Apply the velocity-vector rotation quaternion to cluster data
def trans_clust_vel_GFC(clus_df, rot_2_apply):
    N = len(clus_df)
    new_clus = np.zeros((N,4))
    for row in range(N):
        quat_i = np.array([clus_df.values[row, 0], clus_df.values[row, 1], clus_df.values[row, 2], clus_df.values[row, 3]])
        new_clus[row] = quaternion_multiply(quat_i, rot_2_apply)
    new_clus_df = pd.DataFrame(new_clus)
    return new_clus_df


# Apply the velocity-vector rotation quaternion to cluster data
def trans_clust_vel_LFC(clus_df, rot_2_apply):
    N = len(clus_df)
    new_clus = np.zeros((N,4))
    for row in range(N):
        quat_i = np.array([clus_df.values[row, 0], clus_df.values[row, 1], clus_df.values[row, 2], clus_df.values[row, 3]])
        new_clus[row] = quaternion_multiply(rot_2_apply, quat_i)
    new_clus_df = pd.DataFrame(new_clus)
    return new_clus_df

# Write new data to APDM file template
def write_to_APDM(df_1, df_2, df_3, df_4, template_file, tag):
    # Make columns of zeros
    N = len(df_1)
    zeros_25_df = pd.DataFrame(np.zeros((N, 25)))
    zeros_11_df = pd.DataFrame(np.zeros((N, 11)))
    zeros_2_df = pd.DataFrame(np.zeros((N, 2)))

    # Make a dataframe with zeros columns inbetween the data
    IMU_and_zeros_df = pd.concat([zeros_25_df, df_1, zeros_11_df, df_2, zeros_11_df, df_3, zeros_11_df, df_4, zeros_2_df], axis=1)

    # Read in the APDM template and save as an array
    with open(template_file, 'r') as file:
        template_df = pd.read_csv(file, header=0)
        template_array = template_df.to_numpy()

    # Concatenate the IMU_and_zeros and the APDM template headings
    IMU_and_zeros_array = IMU_and_zeros_df.to_numpy()
    new_array = np.concatenate((template_array, IMU_and_zeros_array), axis=0)
    new_df = pd.DataFrame(new_array)

    # Add the new dataframe into the template
    new_df.to_csv("APDM_" + tag + ".csv", mode='w', index=False, header=False, encoding='utf-8', na_rep='nan')


# Write the transformed IMU data and original cluster data to a .txt file
def write_to_txt(IMU1_df_trans, IMU2_df_trans, IMU3_df_trans, IMU4_df_trans, OMC_Clus_df, tag):
    # Turn data frames into numpy arrays
    OMC_Clus_array = OMC_Clus_df.to_numpy()
    IMU1_array = IMU1_df_trans.to_numpy()
    IMU2_array = IMU2_df_trans.to_numpy()
    IMU3_array = IMU3_df_trans.to_numpy()
    IMU4_array = IMU4_df_trans.to_numpy()
    all_data = np.concatenate((IMU1_array, IMU2_array, IMU3_array, IMU4_array, OMC_Clus_array), axis=1)
    output_file_name = tag + "_transformed.csv"
    # Output data to a csv file
    text_file_header = "IMU1_Q0,IMU1_Q1,IMU1_Q2,IMU1_Q3,IMU2_Q0,IMU2_Q1,IMU2_Q2,IMU2_Q3," \
                       "IMU3_Q0,IMU3_Q1,IMU3_Q2,IMU3_Q3,IMU4_Q0, IMU4_Q1, IMU4_Q2, IMU4_Q3, " \
                       "OpTr_Q0, OpTr_Q1, OpTr_Q2, OpTr_Q3"
    np.savetxt(output_file_name, all_data, delimiter=',', fmt='%.6f', header=text_file_header, comments='')


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
    write_to_txt(IMU1_df_trans, IMU2_df_trans, IMU3_df_trans, OpTr_Clus_df, NewAx_Clus_df, )
    plot_the_quats(IMU1_df_trans, NewAx_Clus_df, tag, sample_rate)
