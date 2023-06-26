import logging
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from pre_process import read_data_frame_from_file
from pre_process import transform_IMU_data
from pre_process import write_to_APDM
from pre_process import write_to_txt
from pre_process import plot_the_quats
from pre_process import interpolate_dfs
from pre_process import trim_df
from pre_process import trans_clust_t0
from pre_process import trans_clust_average
from pre_process import vel_rot_quat_GCF
from pre_process import vel_rot_quat_LCF
from pre_process import trans_clust_vel_GFC
from pre_process import trans_clust_vel_LFC
from analysis import eulers_from_quats
from analysis import plot_four_IMU_eulers
from analysis import plot_IMU_vs_Clus_eulers
from analysis import find_quat_diff
from analysis import find_smallest_angle
from analysis import proj_vec_angle_global_Y
from analysis import plot_proj_vec_angle
from analysis import BA_plot
from analysis import vec_angle_vel
from analysis import BA_plot_eulers
from analysis import BA_plot_combine_reps
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


def full_analysis(input_file, start_time, end_time):

    tag = input_file.replace(" - Report1.txt", "")
    ### SETTINGS

    # Choose outputs
    trim_data = True
    plot_quats = False
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

    # Trim the data based on start and end time
    if trim_data == True:
        IMU1_df_raw = trim_df(IMU1_df_raw, start_time, end_time, sample_rate)
        OpTr_Clus_df_raw = trim_df(OpTr_Clus_df_raw, start_time, end_time, sample_rate)
        NewAx_Clus_df_raw = trim_df(NewAx_Clus_df_raw, start_time, end_time, sample_rate)

    # Interpolate for missing data
    IMU1_df_raw, OpTr_Clus_df_raw, NewAx_Clus_df_raw, total_nans = interpolate_dfs(IMU1_df_raw, OpTr_Clus_df_raw, NewAx_Clus_df_raw)
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


    ### ANALYSE THE DATA

    # Find the 3-angle quaternion-based orientation difference between an IMU and OMC LCF.
    diff_angle_1, diff_angle_2, diff_angle_3, RMSD_angle_1, RMSD_angle_2, RMSD_angle_3 = find_quat_diff(IMU1_df, OMC_Clus_df, ext_decomp_seq)

    # Find the single-angle quaternion-based orientation difference between an IMU and OMC LCF.
    single_angle_diff, RMSD_single_angle = find_smallest_angle(IMU1_df, OMC_Clus_df)

    logging.info(input_file + ": OMC LCF used: " + which_OMC_LCF)
    logging.info(input_file + ": Angle 1 RMSD: " + str(RMSD_angle_1))
    logging.info(input_file + ": Angle 2 RMSD: " + str(RMSD_angle_2))
    logging.info(input_file + ": Angle 3 RMSD: " + str(RMSD_angle_3))
    logging.info(input_file + ": Smallest Angle RMSD: " + str(RMSD_single_angle))


    ### PLOT RESULTS

    # Find the angle of the projected local y vector on the chosen global plane (Cluster LCF):
    proj_vec_angle = proj_vec_angle_global_Y(OMC_Clus_df, global_axis)

    # Calculate Euler angles from the quaternions
    IMU1_eul_1, IMU1_eul_2, IMU1_eul_3 = eulers_from_quats(IMU1_df, int_decomp_seq)
    OMC_Clus_eul_1, OMC_Clus_eul_2, OMC_Clus_eul_3 = eulers_from_quats(OMC_Clus_df, int_decomp_seq)

    # Plot two sets of quaternions for comparison and checking timings (set to stylus-defined cluster and IMU1)
    if plot_quats == True:
        plot_the_quats(IMU1_df, OMC_Clus_df, tag, sample_rate)

    # Plot the Euler angles
    if plot_IMUvsClust_eulers == True:
        plot_IMU_vs_Clus_eulers(IMU1_eul_1, IMU1_eul_2, IMU1_eul_3, OMC_Clus_eul_1, OMC_Clus_eul_2, OMC_Clus_eul_3, single_angle_diff, int_decomp_seq, tag, sample_rate)

    if plot_proj_vector_angle == True:
        plot_proj_vec_angle(proj_vec_angle, sample_rate, global_axis, tag)

    if plot_BA == True:
        BA_plot(proj_vec_angle, single_angle_diff, tag)

    if plot_BA_Eulers == True:
        BA_plot_eulers(IMU1_eul_1, OMC_Clus_eul_1, "Y", tag)

    return RMSD_angle_1, RMSD_angle_2, RMSD_angle_3, RMSD_single_angle, proj_vec_angle, single_angle_diff



### RUN THE ANALYSIS DEFINED ABOVE

# Run the analysis for each rep, returning average orientation differences
R1_RMSD_angle_1, R1_RMSD_angle_2, R1_RMSD_angle_3, R1_RMSD_single_angle, proj_vec_angle_R1, single_angle_diff_R1 = full_analysis(file_label + "_R1 - Report1.txt", start_time = 0, end_time = 30)
R2_RMSD_angle_1, R2_RMSD_angle_2, R2_RMSD_angle_3, R2_RMSD_single_angle, proj_vec_angle_R2, single_angle_diff_R2 = full_analysis(file_label + "_R2 - Report1.txt", start_time = 0, end_time = 30)
R3_RMSD_angle_1, R3_RMSD_angle_2, R3_RMSD_angle_3, R3_RMSD_single_angle, proj_vec_angle_R3, single_angle_diff_R3 = full_analysis(file_label + "_R3 - Report1.txt", start_time = 0, end_time = 30)


### CALCULATE AVERAGES OVER ALL REPS

# Calculate the average RMSD across the five reps, and the deviation in that RMSD
average_RMSD_angle1 = np.mean([R1_RMSD_angle_1, R2_RMSD_angle_1, R3_RMSD_angle_1])
average_SD_angle1 = np.std([R1_RMSD_angle_1, R2_RMSD_angle_1, R3_RMSD_angle_1])
average_RMSD_angle2 = np.mean([R1_RMSD_angle_2, R2_RMSD_angle_2, R3_RMSD_angle_2])
average_SD_angle2 = np.std([R1_RMSD_angle_2, R2_RMSD_angle_2, R3_RMSD_angle_2])
average_RMSD_angle3 = np.mean([R1_RMSD_angle_3, R2_RMSD_angle_3, R3_RMSD_angle_3])
average_SD_angle3 = np.std([R1_RMSD_angle_3, R2_RMSD_angle_3, R3_RMSD_angle_3])
average_RMSD_single_angle = np.mean([R1_RMSD_single_angle, R2_RMSD_single_angle, R3_RMSD_single_angle])
average_SD_single_angle = np.std([R1_RMSD_single_angle, R2_RMSD_single_angle, R3_RMSD_single_angle])

logging.info("\n Intrinsic decomp seq: " + int_decomp_seq + "\n Extrinsic decomp seq: " + ext_decomp_seq)
logging.info("Average Results: \n"
             "Average RMSE - Angle 1: " + str(round(average_RMSD_angle1, 4)) + " SD: " + str(round(average_SD_angle1, 4)) + "\n"
             "Average RMSE - Angle 2: " + str(round(average_RMSD_angle2, 4)) + " SD: " + str(round(average_SD_angle2, 4)) + "\n"
             "Average RMSE - Angle 3: " + str(round(average_RMSD_angle3, 4)) + " SD: " + str(round(average_SD_angle3, 4)) + "\n"
             "Average RMSE - Single-Angle: " + str(round(average_RMSD_single_angle, 4)) + " SD: " + str(round(average_SD_single_angle, 4)))


### BA PLOT ALL DATA COMBINED

BA_plot_combine_reps(proj_vec_angle_R1, single_angle_diff_R1, proj_vec_angle_R2, single_angle_diff_R2, proj_vec_angle_R3, single_angle_diff_R3, file_label)




