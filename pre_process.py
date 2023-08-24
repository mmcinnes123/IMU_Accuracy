### Functions needed to read in the raw data, apply transformations, and write to APDM template file.
    # Raw data is in  MotionMonitor report file
    # APDM template file can be used to visualise quats in OpenSim
    # Transformations:
        # IMU data into Y-Up convention
        # LCF alginment - transform cluster data based on relative orientation to IMU at t=0, or average

from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
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


# Apply some pre-define transformations to create synthetic data
def create_synthetic_data(quat_df, rot_quat_GCF, rot_quat_LCF):
    N = len(quat_df)
    quat_df_transformed = np.zeros((N, 4))
    for row in range(N):
        quat_i = np.array([quat_df.values[row, 0], quat_df.values[row, 1], quat_df.values[row, 2], quat_df.values[row, 3]])
        quat_df_transformed[row] = quat_mul(quat_mul(quat_conj(rot_quat_GCF), quat_i), quat_conj(rot_quat_LCF))
    quat_df_transformed = pd.DataFrame(quat_df_transformed)
    return quat_df_transformed


# Interpolate all the data and return how many missing data points there were.
def interpolate_dfs(IMU1_df_trans, OpTr_Clus_df, NewAx_Clus_df):
    IMU1_df_trans = IMU1_df_trans.interpolate(limit=50)
    OpTr_Clus_df = OpTr_Clus_df.interpolate(limit=50)
    NewAx_Clus_df = NewAx_Clus_df.interpolate(limit=50)
    nan_count1 = IMU1_df_trans.isna().sum().sum()
    nan_count2 = OpTr_Clus_df.isna().sum().sum()
    nan_count3 = NewAx_Clus_df.isna().sum().sum()
    total_nans = nan_count2 + nan_count1 + nan_count3
    return IMU1_df_trans, OpTr_Clus_df, NewAx_Clus_df, total_nans

# Interpolate all the data and return how many missing data points there were.
def interpolate_df(df):
    df = df.interpolate(limit=50)
    nan_count = df.isna().sum().sum()
    return df, nan_count




# Cross-correlation using stattools
def cross_corr_quats(IMU_df, OMC_df):

    # Calculate the cross correlation of each quat element seperately (q0, q1, q2, q3)
    all_corrs = np.zeros((4,len(IMU_df)))
    for i in range(4):
        sig1 = np.array(abs(IMU_df.values[:, i]))
        sig2 = np.array(abs(OMC_df.values[:,i]))
        corr = sm.tsa.stattools.ccf(sig1, sig2, adjusted=False)
        all_corrs[i] = corr

    # Sum the correlation coefficients related to each q-element
    sum_corr = all_corrs[0] + all_corrs[1] + all_corrs[2] + all_corrs[3]

    # Calculate the lag based on the position on the maximum value of the summed correlation coefficients
    lag = np.argmax(sum_corr)

    return lag





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


# Transform cluster quaternions based on a rotation quaternion between IMU and Cluster at t=0
def trans_clust_t0(Clus_df, IMU_df):
    q_clus_t0 = np.array([Clus_df.values[0, 0], Clus_df.values[0, 1], Clus_df.values[0, 2], Clus_df.values[0, 3]])
    q_IMU_t0 = np.array([IMU_df.values[0, 0], IMU_df.values[0, 1], IMU_df.values[0, 2], IMU_df.values[0, 3]])
    q_rot_clust2IMU = quat_mul(quat_conj(q_clus_t0), q_IMU_t0)
    N = len(Clus_df)
    trans_Clus_quats = np.zeros((N, 4))
    for row in range(N):
        clust_quat_i = np.array([Clus_df.values[row, 0], Clus_df.values[row,1], Clus_df.values[row,2], Clus_df.values[row,3]])
        trans_Clus_quats[row] = quat_mul(clust_quat_i, q_rot_clust2IMU)
    transformed_clust_quats = pd.DataFrame(trans_Clus_quats)
    return transformed_clust_quats


# Transform cluster quaternions based on an average rotation quaternion between IMU and Cluster
def trans_clust_average(Clus_df, IMU_df):
    N = len(Clus_df)
    q_rot_clust2IMU = np.zeros((N, 4))
    for row in range(N):
        q_clust = Clus_df.values[row, :]
        q_IMU = IMU_df.values[row, :]
        q_rot_clust2IMU[row] = quat_mul(quat_conj(q_clust), q_IMU)
    q_rot_clust2IMU = q_rot_clust2IMU[~np.isnan(q_rot_clust2IMU).any(axis=1)]
    q_rot_average = average_quaternions(q_rot_clust2IMU)
    trans_Clus_quats = np.zeros((N, 4))
    for row in range(N):
        clust_quat_i = np.array([Clus_df.values[row, 0], Clus_df.values[row,1], Clus_df.values[row,2], Clus_df.values[row,3]])
        trans_Clus_quats[row] = quat_mul(clust_quat_i, q_rot_average)
    transformed_clust_quats = pd.DataFrame(trans_Clus_quats)
    return transformed_clust_quats


# Apply the GCF and LCF rotation quaternion to IMU data
def apply_LCF_and_GCF_to_IMU(quat_df, GCF_trans_quat, LCF_trans_quat):
    N = len(quat_df)
    new_quat = np.zeros((N, 4))
    for row in range(N):
        quat_i = np.array([quat_df.values[row, 0], quat_df.values[row, 1], quat_df.values[row, 2], quat_df.values[row, 3]])
        new_quat[row] = quat_mul(quat_mul(GCF_trans_quat, quat_i), LCF_trans_quat)
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


