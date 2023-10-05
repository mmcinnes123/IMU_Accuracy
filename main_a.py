# This script finds the global misalignment between the IMU and OMC system
# The output is an alignment quaternion which can be applied to all IMU data in further processing
# The inputs are multiple types of movement data

import logging
from pre_process import *


# INITIAL SETTINGS
sample_rate = 100
logging.basicConfig(filename="Results_GCF_Alignment.log", level=logging.INFO)
logging.info("RUNNING FILE: " + str(__file__))


# DEFINE THE FUNCTION

# Define a function which calculates GCF misalignment from an input file
def find_GCF_trans_quat(input_file, start_time, end_time, data_type, which_limb):

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

    # Find the average GCF transformation quaternion with assumption that LCFs are perfectly aligned
    GCF_rot_quat = find_average_GCF_rot_quat(IMU_df, OMC_df_raw)

    return GCF_rot_quat


# USE THE FUNCTION

# Only use first 30s of all tests
start_time = 0
end_time = 30
# Initiate an array to hold all quats
GCF_rot_quats = np.zeros((0, 4))

# Calculate GCF misalignment quaternion from all FUN data
list_of_limbs = ["Torso", "Upper", "Forearm"]
list_of_files = ["FUN_R1 - Report2.txt", "FUN_R2 - Report2.txt", "FUN_R3 - Report2.txt", "FUN_R4 - Report2.txt",
                 "FUN_R5 - Report2.txt"]
for limb in list_of_limbs:
    for file_name in list_of_files:
        GCF_rot_quat = find_GCF_trans_quat(file_name, start_time, end_time, data_type='FUN', which_limb=limb)
        GCF_rot_quats = np.vstack([GCF_rot_quats, GCF_rot_quat])

# Calculate GCF misalignment quaternion from all CON data
list_of_labels = ["CON_MP", "CON_HO", "CON_VE"]
list_of_reps = ["_R1 - Report2.txt", "_R2 - Report2.txt", "_R3 - Report2.txt", "_R4 - Report2.txt",
                "_R5 - Report2.txt"]
for label in list_of_labels:
    for rep in list_of_reps:
        file_name = label + rep
        GCF_rot_quat = find_GCF_trans_quat(file_name, start_time, end_time, data_type='CON', which_limb='NONE')
        GCF_rot_quats = np.vstack([GCF_rot_quats, GCF_rot_quat])


# COMBINE ALL RESULTS

# Calculate the average GCF misalignment quaternion from all the reps/types of movement
GCF_alignment_quat = average_quaternions(GCF_rot_quats)

logging.info("GCF Alignment Quaternion: ")
logging.info(GCF_alignment_quat)


