# This script runs the required pre-processing steps to all the raw data
# Inputs are all the raw data files
# Outputs are .csv files for every rep of all CON and FUN data
    # A file is created for all CON data used for absolute accuracy analysis
    # A file is created for all CON data used for inter-IMU agreement analysis
    # A seperate Torso, Upper, and Forearm file is created for all FUN data used for absolute accuracy analysis
    # A seperate Torso, Upper, and Forearm file is created for all FUN data used for joint angle analysis


import logging
from pre_process import *
import os

logging.basicConfig(filename="Results_PreProcess.log", level=logging.INFO)
logging.info("RUNNING FILE: " + str(__file__))

# Make a folder to store the processed data
new_folder_name = 'ProcessedData'
os.makedirs(new_folder_name, exist_ok=True)


### DEFINE GLOBAL AND LOCAL MISALIGNMENT QUATERNIONS

# Use GCF alignment quaternion calculated in main_a.py
GCF_alignment_quat = np.array([-9.47112458e-01, -4.33103184e-04, 3.20885659e-01, 3.19327766e-03])
logging.info("GCF alignment quaternion: " + str(np.around(GCF_alignment_quat, 4)))

# Use LCF alignment quaternions calculated in main_b.py
LCF_alignment_quat_CON = [0.99875798, -0.01179207,  0.00978239,  0.04741044]
LCF_alignment_quat_IMU2 = [0.99996261, -0.0082542,  -0.00130569,  0.00222294]
LCF_alignment_quat_IMU3 = [9.99919007e-01, -1.25052029e-02, -2.17818831e-03,  9.24452645e-04]
LCF_alignment_quat_IMU4 = [9.99881404e-01, -7.31607851e-03, -1.35508240e-02,  1.66177452e-04]
LCF_alignment_quat_T = [-0.99663601,  0.00436451, -0.0661548,  -0.04817834]
LCF_alignment_quat_U = [0.998934,   0.01345973, 0.02845089, 0.03376764]
LCF_alignment_quat_F = [0.99850962,  0.02547898, -0.00221886,  0.04821239]



### DEFINE FUNCTIONS FOR PRE-PROCESSING ALL DATA

# Define a function for CON data
def pre_process_CON_data(input_file):

    tag = input_file.replace(" - Report2.txt", "")

    # Read data from the file
    IMU1_df_raw, IMU2_df_raw, IMU3_df_raw, IMU4_df_raw, OpTr_Clus_df_raw, NewAx_Clus_df_raw = read_data_frame_from_file((input_file))

    # Interpolate for missing data
    IMU1_df, IMU1_nan_count = interpolate_df(IMU1_df_raw)
    IMU2_df, IMU2_nan_count = interpolate_df(IMU2_df_raw)
    IMU3_df, IMU3_nan_count = interpolate_df(IMU3_df_raw)
    IMU4_df, IMU4_nan_count = interpolate_df(IMU4_df_raw)
    OMC_df, OMC_nan_count = interpolate_df(NewAx_Clus_df_raw)
    total_nans = IMU1_nan_count + IMU2_nan_count + IMU3_nan_count + IMU4_nan_count + OMC_nan_count

    # Do initial transform of IMU data to match OptiTrack Y-up convention
    IMU1_df = intial_IMU_transform(IMU1_df)
    IMU2_df = intial_IMU_transform(IMU2_df)
    IMU3_df = intial_IMU_transform(IMU3_df)
    IMU4_df = intial_IMU_transform(IMU4_df)

    # Apply LCF alignment quaternions to align each IMU with IMU1
    IMU1_df_aligned = IMU1_df
    IMU2_df_aligned = apply_LCF_to_IMU(IMU2_df, LCF_alignment_quat_IMU2)
    IMU3_df_aligned = apply_LCF_to_IMU(IMU3_df, LCF_alignment_quat_IMU3)
    IMU4_df_aligned = apply_LCF_to_IMU(IMU4_df, LCF_alignment_quat_IMU4)

    # Write IMU data to new file for Inter-IMU processing
    all_IMUs_df = pd.concat([IMU1_df_aligned, IMU2_df_aligned, IMU3_df_aligned, IMU4_df_aligned], axis=1)
    all_IMUs_df.columns = ["IMU1_Q0", "IMU1_Q1", "IMU1_Q2", "IMU1_Q3", "IMU2_Q0", "IMU2_Q1", "IMU2_Q2", "IMU2_Q3",
                           "IMU3_Q0", "IMU3_Q1", "IMU3_Q2", "IMU3_Q3", "IMU4_Q0", "IMU4_Q1", "IMU4_Q2", "IMU4_Q3"]
    all_IMUs_df.to_csv(new_folder_name + "/Inter_" + tag + ".csv", header=True, index=True)

    ### Apply transformations to one IMU for absolute comparison with OMC

    # Work with IMU2 since this one was used for stylus-based LCF definition
    IMU_df = IMU2_df

    # Apply the calculated rot_quat LCF and GCF to the IMU data
    IMU_df = apply_LCF_and_GCF_to_IMU(IMU_df, GCF_alignment_quat, LCF_alignment_quat_CON)

    # Write the IMU and OMC data to a new file for Absolute Accuracy processing
    IMU_and_OMC_df = pd.concat([IMU_df, OMC_df], axis=1)
    IMU_and_OMC_df.columns = ["IMU_Q0", "IMU_Q1", "IMU_Q2", "IMU_Q3", "OMC_Q0", "OMC_Q1", "OMC_Q2", "OMC_Q3"]
    IMU_and_OMC_df.to_csv(new_folder_name + "/" + tag + ".csv", header=True, index=True)

    logging.info(str(tag))
    logging.info("No of missing samples: " + str(total_nans))


# Define a function for FUN data
def pre_process_FUN_data(input_file):

    tag = input_file.replace(" - Report2.txt", "")

    ### READ DATA IN

    # Read data from the file
    IMU_T_df_raw, IMU_U_df_raw, IMU_F_df_raw, OpTr_T_Clus_df, NewAx_T_Clus_df, OpTr_U_Clus_df, NewAx_U_Clus_df, OpTr_F_Clus_df, NewAx_F_Clus_df \
        = read_data_frame_from_file_FUN((input_file))

    ### APPLY INITIAL CHANGES

    # Interpolate for missing data
    IMU_T_df, IMU_T_nan_count = interpolate_df(IMU_T_df_raw)
    IMU_U_df, IMU_U_nan_count = interpolate_df(IMU_U_df_raw)
    IMU_F_df, IMU_F_nan_count = interpolate_df(IMU_F_df_raw)
    OMC_T_df, OMC_T_nan_count = interpolate_df(NewAx_T_Clus_df)
    OMC_U_df, OMC_U_nan_count = interpolate_df(NewAx_U_Clus_df)
    OMC_F_df, OMC_F_nan_count = interpolate_df(NewAx_F_Clus_df)
    total_IMU_nans = IMU_T_nan_count + IMU_U_nan_count + IMU_F_nan_count
    total_OMC_nans = OMC_T_nan_count + OMC_U_nan_count + OMC_F_nan_count

    # Do initial transform of IMU data to match OptiTrack Y-up convention
    IMU_T_df = intial_IMU_transform(IMU_T_df)
    IMU_U_df = intial_IMU_transform(IMU_U_df)
    IMU_F_df = intial_IMU_transform(IMU_F_df)

    ### WRITE TO NEW FILE FOR JOINT ANGLE ANALYSIS

    # Write the Torso IMU and OMC data to a new file for further processing
    new_header_list = ["IMU_Q0", "IMU_Q1", "IMU_Q2", "IMU_Q3", "OMC_Q0", "OMC_Q1", "OMC_Q2", "OMC_Q3"]
    IMU_and_OMC_T_df = pd.concat([IMU_T_df, OMC_T_df], axis=1)
    IMU_and_OMC_T_df.columns = new_header_list
    IMU_and_OMC_T_df.to_csv(new_folder_name + "/" + "JA_Torso_" + tag + ".csv", header=True, index=True)

    # Write the Upper arm IMU and OMC data to a new file for further processing
    IMU_and_OMC_U_df = pd.concat([IMU_U_df, OMC_U_df], axis=1)
    IMU_and_OMC_U_df.columns = new_header_list
    IMU_and_OMC_U_df.to_csv(new_folder_name + "/" + "JA_Upper_" + tag + ".csv", header=True, index=True)

    # Write the Forearm arm IMU and OMC data to a new file for further processing
    IMU_and_OMC_F_df = pd.concat([IMU_F_df, OMC_F_df], axis=1)
    IMU_and_OMC_F_df.columns = new_header_list
    IMU_and_OMC_F_df.to_csv(new_folder_name + "/" + "JA_Forearm_" + tag + ".csv", header=True, index=True)

    ### ALIGN IMU FRAME WITH OMC FOR ABSOLUTE ACCURACY ANALYSIS

    # Apply the calculated rot_quat LCF and GCF to the IMU data
    IMU_T_df = apply_LCF_and_GCF_to_IMU(IMU_T_df, GCF_alignment_quat, LCF_alignment_quat_T)
    IMU_U_df = apply_LCF_and_GCF_to_IMU(IMU_U_df, GCF_alignment_quat, LCF_alignment_quat_U)
    IMU_F_df = apply_LCF_and_GCF_to_IMU(IMU_F_df, GCF_alignment_quat, LCF_alignment_quat_F)

    ### WRITE TO NEW FILE FOR ABSOLUTE ACCURACY ANALYSIS

    # Write the Torso IMU and OMC data to a new file for further processing
    new_header_list = ["IMU_Q0", "IMU_Q1", "IMU_Q2", "IMU_Q3", "OMC_Q0", "OMC_Q1", "OMC_Q2", "OMC_Q3"]
    IMU_and_OMC_T_df = pd.concat([IMU_T_df, OMC_T_df], axis=1)
    IMU_and_OMC_T_df.columns = new_header_list
    IMU_and_OMC_T_df.to_csv(new_folder_name + "/" + "Torso_" + tag + ".csv", header=True, index=True)

    # Write the Upper arm IMU and OMC data to a new file for further processing
    IMU_and_OMC_U_df = pd.concat([IMU_U_df, OMC_U_df], axis=1)
    IMU_and_OMC_U_df.columns = new_header_list
    IMU_and_OMC_U_df.to_csv(new_folder_name + "/" + "Upper_" + tag + ".csv", header=True, index=True)

    # Write the Forearm arm IMU and OMC data to a new file for further processing
    IMU_and_OMC_F_df = pd.concat([IMU_F_df, OMC_F_df], axis=1)
    IMU_and_OMC_F_df.columns = new_header_list
    IMU_and_OMC_F_df.to_csv(new_folder_name + "/" + "Forearm_" + tag + ".csv", header=True, index=True)

    logging.info(str(tag))
    logging.info("No of missing IMU samples: " + str(total_IMU_nans))
    logging.info("No of missing OMC samples: " + str(total_OMC_nans))



### RUN FUNCTION FOR ALL REPS OF CON DATA

# 5 reps of multi-plane, horizontal RoM, and vertical RoM movements
list_of_labels = ["CON_MP", "CON_HO", "CON_VE"]
for label in list_of_labels:
    for i in range(1,6):
        file_name = label + "_R" + str(i) + " - Report2.txt"
        pre_process_CON_data(file_name)
# 3 reps of the static placements
for i in range(1,4):
    file_name = "CON_SP" + "_R" + str(i) + " - Report2.txt"
    pre_process_CON_data(file_name)



## RUN FUNCTION FOR ALL REPS OF FUN DATA

# 5 reps of the functional upper-limb movements
for i in range(1,6,1):
    file_name = "FUN_R" + str(i) + " - Report2.txt"
    pre_process_FUN_data(file_name)



### ADDITIONAL FUNCTION: FOR VISUALISATION

# This code can be added in if any data needs to be visualised
# The code writes the quaternions into a template file which can be read into OpenSim
# Make sure APDM template file template is in same folder, with numbers as first row

    # if write_to_APDM == True:
    #     APDM_template_file = "APDM_template_4S.csv"
    #     write_to_APDM(df_1, df_2, df_3, df_4, APDM_template_file, tag)

# Once data has been written to file, use these commands in terminal, using an environment with OpenSim installed:

# opensense -ReadAPDM APDM_template_4S.csv APDM_Settings_4S.xml
# (Make sure APDM Settings file is in same folder)

# Use 'Preview Sensor Data' option in OpenSim GUI to visualise the movement
