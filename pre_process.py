### Functions needed to read in the raw data, apply transformations, and write to APDM template file.
    # Raw data is in  MotionMonitor report file
    # APDM template file can be used to visualise quats in OpenSim
    # Transformations:
        # IMU data into Y-Up convention
        # LCF alginment - transform cluster data based on relative orientation to IMU at t=0, or average

from scipy.spatial.transform import Rotation as R
import pandas as pd
from quat_functions import *


# Read all data in from specified input file
def read_data_frame_from_file(input_file):
    with open("RawData/" + input_file, 'r') as file:
        df = pd.read_csv(file, header=5, sep="\t")
    # Make seperate data frames
    IMU1_df = df.filter(["IMU1_Q0", "IMU1_Q1", "IMU1_Q2", "IMU1_Q3"], axis=1)
    IMU2_df = df.filter(["IMU2_Q0", "IMU2_Q1", "IMU2_Q2", "IMU2_Q3"], axis=1)
    IMU3_df = df.filter(["IMU3_Q0", "IMU3_Q1", "IMU3_Q2", "IMU3_Q3"], axis=1)
    IMU4_df = df.filter(["IMU4_Q0", "IMU4_Q1", "IMU4_Q2", "IMU4_Q3"], axis=1)
    OpTr_Clus_df = df.filter(["OpTr_Q0", "OpTr_Q1", "OpTr_Q2", "OpTr_Q3"], axis=1)
    NewAx_Clus_df = df.filter(["NewAx_Q0", "NewAx_Q1", "NewAx_Q2", "NewAx_Q3"], axis=1)
    return IMU1_df, IMU2_df, IMU3_df, IMU4_df, OpTr_Clus_df, NewAx_Clus_df


# Read all data in from specified input file (FUNCTIONAL data)
def read_data_frame_from_file_FUN(input_file):
    with open("RawData/" + input_file, 'r') as file:
        df = pd.read_csv(file, header=5, sep="\t")
    # Make seperate data frames
    IMU1_df = df.filter(["IMU1_Q0", "IMU1_Q1", "IMU1_Q2", "IMU1_Q3"], axis=1)
    IMU2_df = df.filter(["IMU2_Q0", "IMU2_Q1", "IMU2_Q2", "IMU2_Q3"], axis=1)
    IMU3_df = df.filter(["IMU3_Q0", "IMU3_Q1", "IMU3_Q2", "IMU3_Q3"], axis=1)
    OpTr_T_Clus_df = df.filter(["OpTr_T_Q0", "OpTr_T_Q1", "OpTr_T_Q2", "OpTr_T_Q3"], axis=1)
    NewAx_T_Clus_df = df.filter(["NewAx_T_Q0", "NewAx_T_Q1", "NewAx_T_Q2", "NewAx_T_Q3"], axis=1)
    OpTr_U_Clus_df = df.filter(["OpTr_U_Q0", "OpTr_U_Q1", "OpTr_U_Q2", "OpTr_U_Q3"], axis=1)
    NewAx_U_Clus_df = df.filter(["NewAx_U_Q0", "NewAx_U_Q1", "NewAx_U_Q2", "NewAx_U_Q3"], axis=1)
    OpTr_F_Clus_df = df.filter(["OpTr_F_Q0", "OpTr_F_Q1", "OpTr_F_Q2", "OpTr_F_Q3"], axis=1)
    NewAx_F_Clus_df = df.filter(["NewAx_F_Q0", "NewAx_F_Q1", "NewAx_F_Q2", "NewAx_F_Q3"], axis=1)
    return IMU1_df, IMU2_df, IMU3_df, OpTr_T_Clus_df, NewAx_T_Clus_df, OpTr_U_Clus_df, NewAx_U_Clus_df, OpTr_F_Clus_df, NewAx_F_Clus_df


# Interpolate all the data and return how many missing data points there were.
def interpolate_df(df):
    df = df.interpolate(limit=50)
    nan_count = df.isna().sum().sum()
    return df, nan_count


# Trim the data frames based on start and end time
def trim_df(df, start_time, end_time, sample_rate):
    first_index = int(start_time*sample_rate)
    last_index = int(end_time*sample_rate)
    index_range = list(range(first_index, last_index))
    df_new = df.iloc[index_range, :]
    df_new_new = df_new.reset_index(drop=True)
    return df_new_new


# Cuts a dataframe into four sections of interest based on start and end times
def cut_four_sections(df, s1, e1, s2, e2, s3, e3, s4, e4, sample_rate):
    index_range_1 = list(range(int(s1*sample_rate), int(e1*sample_rate)))
    index_range_2 = list(range(int(s2*sample_rate), int(e2*sample_rate)))
    index_range_3 = list(range(int(s3*sample_rate), int(e3*sample_rate)))
    index_range_4 = list(range(int(s4*sample_rate), int(e4*sample_rate)))
    df_1 = df.iloc[index_range_1, :]
    df_2 = df.iloc[index_range_2, :]
    df_3 = df.iloc[index_range_3, :]
    df_4 = df.iloc[index_range_4, :]
    final_df = pd.concat([df_1, df_2, df_3, df_4], ignore_index=True, sort=False)
    return final_df


# Transform IMU data into Y-up convention (and apply transpose so orientations are in global frame, not local)
def intial_IMU_transform(IMU_df):
    header = IMU_df.columns
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
        transformed_quats[row] = quat_mul(rot_quat, quat_i)
    transformed_quats_df = pd.DataFrame(transformed_quats, columns=header)
    return transformed_quats_df


# Find the local coordinate frame transformation quaternion
def find_LCF_trans_quat(IMU_df, Clust_df, GCF_trans_quat):
    N = len(IMU_df)
    LCF_rot_quats = np.zeros((N, 4))
    for i in range(N):
        IMU_quat_i = IMU_df.values[i]
        Clust_quat_i = Clust_df.values[i]
        LCF_rot_quat_i = quat_mul(quat_conj(IMU_quat_i), quat_mul(quat_conj(GCF_trans_quat), Clust_quat_i))
        LCF_rot_quats[i] = LCF_rot_quat_i
    LCF_rot_quat_average = average_quaternions(LCF_rot_quats)

    return LCF_rot_quat_average


# Find the average global coordinate frame transformation quaternion based on one set of IMU + cluster data
def find_average_GCF_rot_quat(IMU_df, Clust_df):
    N = len(IMU_df)
    GCF_rot_quats = np.zeros((N, 4))
    for i in range(N):
        IMU_quat_i = IMU_df.values[0]
        Clust_quat_i = Clust_df.values[0]
        GCF_rot_quat_i = quat_mul(Clust_quat_i, quat_conj(IMU_quat_i))
        GCF_rot_quats[i] = GCF_rot_quat_i

    GCF_rot_quat_average = average_quaternions(GCF_rot_quats)

    return GCF_rot_quat_average


# Apply the GCF and LCF rotation quaternion to IMU data
def apply_LCF_and_GCF_to_IMU(quat_df, GCF_trans_quat, LCF_trans_quat):
    N = len(quat_df)
    new_quat = np.zeros((N, 4))
    for row in range(N):
        quat_i = np.array([quat_df.values[row, 0], quat_df.values[row, 1], quat_df.values[row, 2], quat_df.values[row, 3]])
        new_quat[row] = quat_mul(quat_mul(GCF_trans_quat, quat_i), LCF_trans_quat)
    new_quat_df = pd.DataFrame(new_quat, columns=["IMU_Q0", "IMU_Q1", "IMU_Q2", "IMU_Q3"])
    return new_quat_df


# Apply only the LCF rotation quaternion to IMU data
def apply_LCF_to_IMU(quat_df, LCF_trans_quat):
    N = len(quat_df)
    new_quat = np.zeros((N, 4))
    for row in range(N):
        quat_i = np.array([quat_df.values[row, 0], quat_df.values[row, 1], quat_df.values[row, 2], quat_df.values[row, 3]])
        new_quat[row] = quat_mul(quat_i, LCF_trans_quat)
    new_quat_df = pd.DataFrame(new_quat, columns=["IMU_Q0", "IMU_Q1", "IMU_Q2", "IMU_Q3"])
    return new_quat_df


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



