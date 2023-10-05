# This script finds the global misalignment between the IMU and OMC system
# The output is an alignment quaternion which can be applied to all IMU data in further processing
# The input is multiple types of movement data

import logging
from pre_process import *


# INITIAL SETTINGS
logging.basicConfig(filename="Results_LCF_Alignment.log", level=logging.INFO)
logging.info("RUNNING FILE: " + str(__file__))

# DEFINE GLOBAL MISALIGNMENT QUATERNION

# Use GCF alignment quaternion calculated in main_a.py
GCF_alignment_quat = np.array([-9.47112458e-01, -4.33103184e-04, 3.20885659e-01, 3.19327766e-03])
logging.info("GCF alignment quaternion: " + str(np.around(GCF_alignment_quat, 4)))

sample_rate = 100


# CALCULATE LOCAL FRAME MISALIGNMENT

# Define a function which calculates GCF misalignment from an input file
def find_LCF_rot_quat_abs(input_file, start_time, end_time, data_type, which_limb):

    if data_type == 'FUN':
        # Read data from the file
        IMU1_df, IMU2_df, IMU3_df, OpTr_T_Clus_df, NewAx_T_Clus_df, OpTr_U_Clus_df, NewAx_U_Clus_df, OpTr_F_Clus_df, NewAx_F_Clus_df \
            = read_data_frame_from_file_FUN((input_file))
        if which_limb == "Torso":
            IMU_df_raw = IMU1_df
            OMC_df_raw = NewAx_T_Clus_df
        if which_limb == "Upper":
            IMU_df_raw = IMU2_df
            OMC_df_raw = NewAx_U_Clus_df
        if which_limb == "Forearm":
            IMU_df_raw = IMU3_df
            OMC_df_raw = NewAx_F_Clus_df

    elif data_type == 'CON':
        # Read data from the file
        IMU1_df, IMU2_df, IMU3_df, IMU4_df, OpTr_Clus_df_raw, NewAx_Clus_df_raw = read_data_frame_from_file((input_file))
        # Work with IMU2 since this one was used for stylus-based LCF definition
        IMU_df_raw = IMU2_df
        OMC_df_raw = NewAx_Clus_df_raw


    # Trim the data based on start and end time
    IMU_df_raw = trim_df(IMU_df_raw, start_time, end_time, sample_rate)
    OMC_df_raw = trim_df(OMC_df_raw, start_time, end_time, sample_rate)

    # Interpolate for missing data
    IMU_df_raw, IMU_nan_count = interpolate_df(IMU_df_raw)
    OMC_df_raw, OMC_nan_count = interpolate_df(OMC_df_raw)

    # Do initial transform of IMU data to match OptiTrack Y-up convention, and take transpose
    IMU_df = intial_IMU_transform(IMU_df_raw)

    LCF_alignment_quat = find_LCF_trans_quat(IMU_df, OMC_df_raw, GCF_alignment_quat)


    return LCF_alignment_quat


def find_LCF_rot_quat_inter(input_file, start_time, end_time):

    # Read data from the file
    IMU1_df, IMU2_df, IMU3_df, IMU4_df, OpTr_Clus_df_raw, NewAx_Clus_df_raw = read_data_frame_from_file((input_file))

    # Trim the data based on start and end time
    IMU1_df = trim_df(IMU1_df, start_time, end_time, sample_rate)
    IMU2_df = trim_df(IMU2_df, start_time, end_time, sample_rate)
    IMU3_df = trim_df(IMU3_df, start_time, end_time, sample_rate)
    IMU4_df = trim_df(IMU4_df, start_time, end_time, sample_rate)

    # Interpolate for missing data
    IMU1_df, IMU1_nan_count = interpolate_df(IMU1_df)
    IMU2_df, IMU2_nan_count = interpolate_df(IMU2_df)
    IMU3_df, IMU3_nan_count = interpolate_df(IMU3_df)
    IMU4_df, IMU4_nan_count = interpolate_df(IMU4_df)

    # Do initial transform of IMU data to match OptiTrack Y-up convention
    IMU1_df = intial_IMU_transform(IMU1_df)
    IMU2_df = intial_IMU_transform(IMU2_df)
    IMU3_df = intial_IMU_transform(IMU3_df)
    IMU4_df = intial_IMU_transform(IMU4_df)

    # Find the LCF misalignment of each IMU relative to IMU1 (with no global adjustment)
    GCF_null = np.array([1, 0, 0, 0])
    LCF_alignment_quat_IMU2 = find_LCF_trans_quat(IMU2_df, IMU1_df, GCF_null)
    LCF_alignment_quat_IMU3 = find_LCF_trans_quat(IMU3_df, IMU1_df, GCF_null)
    LCF_alignment_quat_IMU4 = find_LCF_trans_quat(IMU4_df, IMU1_df, GCF_null)

    return LCF_alignment_quat_IMU2, LCF_alignment_quat_IMU3, LCF_alignment_quat_IMU4



# CONTROLLED MOVEMENTS: Calculate misalignment between IMU2 and cluster on test rig using CON_MP data

CON_LCF_rot_quats = np.zeros((0, 4))
Inter_IMU2_LCF_rot_quats = np.zeros((0, 4))
Inter_IMU3_LCF_rot_quats = np.zeros((0, 4))
Inter_IMU4_LCF_rot_quats = np.zeros((0, 4))
start_time = 0
end_time = 30
data_type = "CON"
which_limb = "None"

list_of_reps = ["CON_MP_R1 - Report2.txt", "CON_MP_R2 - Report2.txt", "CON_MP_R3 - Report2.txt",
                "CON_MP_R4 - Report2.txt", "CON_MP_R5 - Report2.txt"]
for rep in list_of_reps:
    input_file = rep
    LCF_rot_quat = find_LCF_rot_quat_abs(input_file, start_time, end_time, data_type, which_limb)
    LCF_alignment_quat_IMU2, LCF_alignment_quat_IMU3, LCF_alignment_quat_IMU4 = find_LCF_rot_quat_inter(input_file,
                                                                                                        start_time,
                                                                                                        end_time)
    CON_LCF_rot_quats = np.vstack([CON_LCF_rot_quats, LCF_rot_quat])
    Inter_IMU2_LCF_rot_quats = np.vstack([Inter_IMU2_LCF_rot_quats, LCF_alignment_quat_IMU2])
    Inter_IMU3_LCF_rot_quats = np.vstack([Inter_IMU3_LCF_rot_quats, LCF_alignment_quat_IMU3])
    Inter_IMU4_LCF_rot_quats = np.vstack([Inter_IMU4_LCF_rot_quats, LCF_alignment_quat_IMU4])

# Calculate the average LCF misalignment quaternion from all the reps
CON_LCF_alignment_quat = average_quaternions(CON_LCF_rot_quats)
Inter_IMU2_LCF_alignment_quat = average_quaternions(Inter_IMU2_LCF_rot_quats)
Inter_IMU3_LCF_alignment_quat = average_quaternions(Inter_IMU3_LCF_rot_quats)
Inter_IMU4_LCF_alignment_quat = average_quaternions(Inter_IMU4_LCF_rot_quats)

logging.info(data_type + " LCF Alignment Quaternion: ")
logging.info(CON_LCF_alignment_quat)
logging.info("IMU2 to IMU1 LCF Alignment Quaternion: ")
logging.info(Inter_IMU2_LCF_alignment_quat)
logging.info("IMU3 to IMU1 LCF Alignment Quaternion: ")
logging.info(Inter_IMU3_LCF_alignment_quat)
logging.info("IMU4 to IMU1 LCF Alignment Quaternion: ")
logging.info(Inter_IMU4_LCF_alignment_quat)


# TORSO: Calculate misalignment between IMU and cluster on Torso

Torso_LCF_rot_quats = np.zeros((0, 4))
start_time = 0
end_time = 300
data_type = "FUN"
which_limb = "Torso"

list_of_reps = ["FUN_R1 - Report2.txt", "FUN_R2 - Report2.txt", "FUN_R3 - Report2.txt",
                "FUN_R4 - Report2.txt", "FUN_R5 - Report2.txt"]
for rep in list_of_reps:
    input_file = rep
    LCF_rot_quat = find_LCF_rot_quat_abs(input_file, start_time, end_time, data_type, which_limb)
    Torso_LCF_rot_quats = np.vstack([Torso_LCF_rot_quats, LCF_rot_quat])

# Calculate the average LCF misalignment quaternion from all the reps
CON_LCF_alignment_quat = average_quaternions(Torso_LCF_rot_quats)

logging.info(which_limb + " LCF Alignment Quaternion: ")
logging.info(CON_LCF_alignment_quat)


# UPPER: Calculate misalignment between IMU and cluster on Upper

Upper_LCF_rot_quats = np.zeros((0, 4))
start_time = 0
end_time = 300
data_type = "FUN"
which_limb = "Upper"

list_of_reps = ["FUN_R1 - Report2.txt", "FUN_R2 - Report2.txt", "FUN_R3 - Report2.txt",
                "FUN_R4 - Report2.txt", "FUN_R5 - Report2.txt"]
for rep in list_of_reps:
    input_file = rep
    LCF_rot_quat = find_LCF_rot_quat_abs(input_file, start_time, end_time, data_type, which_limb)
    Upper_LCF_rot_quats = np.vstack([Upper_LCF_rot_quats, LCF_rot_quat])

# Calculate the average LCF misalignment quaternion from all the reps
CON_LCF_alignment_quat = average_quaternions(Upper_LCF_rot_quats)

logging.info(which_limb + " LCF Alignment Quaternion: ")
logging.info(CON_LCF_alignment_quat)


# FOREARM: Calculate misalignment between IMU and cluster on Forearm

Forearm_LCF_rot_quats = np.zeros((0, 4))
start_time = 0
end_time = 300
data_type = "FUN"
which_limb = "Forearm"

list_of_reps = ["FUN_R1 - Report2.txt", "FUN_R2 - Report2.txt", "FUN_R3 - Report2.txt",
                "FUN_R4 - Report2.txt", "FUN_R5 - Report2.txt"]
for rep in list_of_reps:
    input_file = rep
    LCF_rot_quat = find_LCF_rot_quat_abs(input_file, start_time, end_time, data_type, which_limb)
    Forearm_LCF_rot_quats = np.vstack([Forearm_LCF_rot_quats, LCF_rot_quat])

# Calculate the average LCF misalignment quaternion from all the reps
CON_LCF_alignment_quat = average_quaternions(Forearm_LCF_rot_quats)

logging.info(which_limb + " LCF Alignment Quaternion: ")
logging.info(CON_LCF_alignment_quat)



