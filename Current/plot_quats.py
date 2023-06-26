# Script for quick review of movement - to  plot the quaternions or write to APDM file format to view in OpenSim.

import logging
import pandas as pd
from pre_process import read_data_frame_from_file
from pre_process import transform_IMU_data
from pre_process import write_to_APDM
from pre_process import plot_the_quats
from pre_process import interpolate_dfs
from pre_process import trans_clust_t0
from pre_process import trans_clust_average
from pre_process import vel_rot_quat_LCF
from pre_process import trans_clust_vel_GFC
from pre_process import trans_clust_vel_LFC
from quat_functions import average_quaternions

# Make sure APDM file template is in same folder, with numbers as first row,
# and frequency set to match the recording freq (100Hz)


# SETTINGS
file_label = "MP_F_LT"
APDM_template_file = "APDM_template_4S.csv"
sample_rate = 100
int_decomp_seq = "YXZ"  # Intrinsic decomposition seq - used for plotting euler angles
ext_decomp_seq = "yxz"  # Extrinisic decomposition seq - used for calculating quaternion difference

logging.basicConfig(filename="Results_" + file_label + ".log", level=logging.INFO)


def full_analysis(input_file):
    tag = input_file.replace(" - Report1.txt", "")
    ### SETTINGS

    # Choose outputs
    plot_quats = True
    write_APDM = True
    write_text_file = False
    plot_IMUvsClust_eulers = False
    plot_fourIMUs_eulers = False
    plot_proj_vector_angle = False
    plot_BA = False
    plot_BA_Eulers = False

    # Choose global axis of interest for vector angle projection
    global_axis = "X"
    # Choose which OMC LCF: "NewAx" "t0" or "average" or "vel" or 'local'
    which_OMC_LCF = "local"

    ### TRANSFORM THE IMU DATA

    # Read data from the file
    IMU1_df_raw, OpTr_Clus_df_raw, NewAx_Clus_df_raw = read_data_frame_from_file(input_file)

    # Interpolate for missing data
    IMU1_df_raw, OpTr_Clus_df_raw, NewAx_Clus_df_raw, total_nans = interpolate_dfs(IMU1_df_raw, OpTr_Clus_df_raw,
                                                                                   NewAx_Clus_df_raw)
    print("Total missing data points (nans): " + str(total_nans))

    # Transform the IMU data
    IMU1_df = transform_IMU_data(IMU1_df_raw)

    ### TRANSFORM THE CLUSTER DATA

    # Calculate the neccessary velocity-based rotation vector from Cluster to IMU
    rot_2_apply_arr = vel_rot_quat_LCF(IMU1_df, OpTr_Clus_df_raw)
    rot_2_apply_df = pd.DataFrame(rot_2_apply_arr)
    rot_2_appy = average_quaternions(rot_2_apply_arr)
    print(rot_2_apply_arr[:10])
    print(rot_2_apply_arr.shape)
    print(rot_2_appy)

    # Apply the velocity-based rotation quaternion to the cluster data (based on global)
    OpTr_Clus_df_corr = trans_clust_vel_GFC(OpTr_Clus_df_raw, rot_2_appy)

    # Find the rotation quaternion from OptiTrack's cluster LCF to IMU LCF - at t = 0s
    if which_OMC_LCF == "t0":
        OMC_Clus_df = trans_clust_t0(OpTr_Clus_df_corr, IMU1_df)
    elif which_OMC_LCF == "average":
        OMC_Clus_df = trans_clust_average(OpTr_Clus_df_corr, IMU1_df)
    elif which_OMC_LCF == "local":
        # Apply the velocity-based rotation quaternion to the cluster data (based on local)
        OMC_Clus_df = trans_clust_vel_LFC(OpTr_Clus_df_raw, rot_2_appy)
    else:
        OMC_Clus_df = NewAx_Clus_df_raw

    ### WRITE DATA TO APDM FILE FORMAT

    # Write the transformed IMU data (ONLY 3/4 IMUS) and original cluster data to an APDM file
    if write_APDM == True:
        write_to_APDM(IMU1_df, rot_2_apply_df, NewAx_Clus_df_raw, OMC_Clus_df, APDM_template_file, tag)

    # Plot two sets of quaternions for comparison and checking timings (set to stylus-defined cluster and IMU1)
    if plot_quats == True:
        plot_the_quats(IMU1_df, OMC_Clus_df, tag, sample_rate)

# Plot the quaternions to find start and end times
for i in [1, 2, 3, 4, 5]:
    file_name = file_label + "_R" + str(i) + " - Report2.txt"
    full_analysis(file_name)

