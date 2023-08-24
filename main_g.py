# This script is used to analyse joint angles from FUN data

import logging
from analysis import *

sample_rate = 100
logging.basicConfig(filename="Results_JointAngles.log", level=logging.INFO)
logging.info("RUNNING FILE: " + str(__file__))


def trim_data(input_file, start_time, end_time):

    # Read in the pre-processed data
    IMU_df, OMC_df = read_abs_pre_processed_data_from_file(input_file)

    # Trim the data based on start and end time
    IMU_df = trim_df(IMU_df, start_time, end_time, sample_rate)
    OMC_df = trim_df(OMC_df, start_time, end_time, sample_rate)

    return IMU_df, OMC_df

def calculate_joint_angles(Torso_IMU_df, Torso_OMC_df, Upper_IMU_df, Upper_OMC_df, Forearm_IMU_df, Forearm_OMC_df):


    # Rotate the thorax LCF by 90deg around local y to align with ISB def
    Torso_IMU_df = rotate_thorax(Torso_IMU_df)
    Torso_OMC_df = rotate_thorax(Torso_OMC_df)

    # Specify decomposition sequences
    elbow_decomp_seq = "ZXY"
    shoulder_decomp_seq = "YXY"

    # Calculate joint rotation quaternions
    IMU_q_elbow = find_joint_rot_quat(Upper_IMU_df, Forearm_IMU_df)
    OMC_q_elbow = find_joint_rot_quat(Upper_OMC_df, Forearm_OMC_df)
    IMU_q_shoulder = find_joint_rot_quat(Torso_IMU_df, Upper_IMU_df)
    OMC_q_shoulder = find_joint_rot_quat(Torso_OMC_df, Upper_OMC_df)

    # Calculate 3 euler angles
    IMU_elbow_angle1, IMU_elbow_angle2, IMU_elbow_angle3 = eulers_from_quats(IMU_q_elbow, elbow_decomp_seq)
    OMC_elbow_angle1, OMC_elbow_angle2, OMC_elbow_angle3 = eulers_from_quats(OMC_q_elbow, elbow_decomp_seq)
    IMU_shoulder_angle1, IMU_shoulder_angle2, IMU_shoulder_angle3 = eulers_from_quats(IMU_q_shoulder, shoulder_decomp_seq)
    OMC_shoulder_angle1, OMC_shoulder_angle2, OMC_shoulder_angle3 = eulers_from_quats(OMC_q_shoulder, shoulder_decomp_seq)

    # Applying the following corrections allows the angles to be expressed in same convention as Pearl paper
    IMU_shoulder_angle1 = adjust_angles_by_180(IMU_shoulder_angle1)
    OMC_shoulder_angle1 = adjust_angles_by_180(OMC_shoulder_angle1)
    IMU_shoulder_angle3 = adjust_angles_by_180(IMU_shoulder_angle3)
    OMC_shoulder_angle3 = adjust_angles_by_180(OMC_shoulder_angle3)

    return IMU_elbow_angle1, IMU_elbow_angle2, IMU_elbow_angle3, OMC_elbow_angle1, OMC_elbow_angle2, OMC_elbow_angle3, \
        IMU_shoulder_angle1, IMU_shoulder_angle2, IMU_shoulder_angle3, OMC_shoulder_angle1, OMC_shoulder_angle2, OMC_shoulder_angle3


def find_joint_angle_error(IMU_elbow_angle1, IMU_elbow_angle2, IMU_elbow_angle3, OMC_elbow_angle1, OMC_elbow_angle2, OMC_elbow_angle3, \
            IMU_shoulder_angle1, IMU_shoulder_angle2, IMU_shoulder_angle3, OMC_shoulder_angle1, OMC_shoulder_angle2, OMC_shoulder_angle3):

    RMSE_elbow_angle_1 = find_euler_error(IMU_elbow_angle1, OMC_elbow_angle1)
    RMSE_elbow_angle_2 = find_euler_error(IMU_elbow_angle2, OMC_elbow_angle2)
    RMSE_elbow_angle_3 = find_euler_error(IMU_elbow_angle3, OMC_elbow_angle3)

    RMSE_shoulder_angle_1 = find_euler_error(IMU_shoulder_angle1, OMC_shoulder_angle1)
    RMSE_shoulder_angle_2 = find_euler_error(IMU_shoulder_angle2, OMC_shoulder_angle2)
    RMSE_shoulder_angle_3 = find_euler_error(IMU_shoulder_angle3, OMC_shoulder_angle3)

    return RMSE_elbow_angle_1, RMSE_elbow_angle_2, RMSE_elbow_angle_3, RMSE_shoulder_angle_1, RMSE_shoulder_angle_2, RMSE_shoulder_angle_3



def combine_reps(label):

    # Initiate arrays to hold a single value for each rep
    elbow_angle_1_RMSEs = np.zeros((no_reps))
    elbow_angle_2_RMSEs = np.zeros((no_reps))
    elbow_angle_3_RMSEs = np.zeros((no_reps))
    shoulder_angle_1_RMSEs = np.zeros((no_reps))
    shoulder_angle_2_RMSEs = np.zeros((no_reps))
    shoulder_angle_3_RMSEs = np.zeros((no_reps))

    for i in range(1, no_reps + 1):

        torso_file_name = "JA_Torso_FUN_R" + str(i) + ".csv"
        upper_file_name = "JA_Upper_FUN_R" + str(i) + ".csv"
        forearm_file_name = "JA_Forearm_FUN_R" + str(i) + ".csv"
        tag = label + "_R" + str(i)

        Torso_IMU_df, Torso_OMC_df = trim_data(torso_file_name, start_time, end_time)
        Upper_IMU_df, Upper_OMC_df = trim_data(upper_file_name, start_time, end_time)
        Forearm_IMU_df, Forearm_OMC_df = trim_data(forearm_file_name, start_time, end_time)

        # Calculate the joint angles
        IMU_elbow_angle1, IMU_elbow_angle2, IMU_elbow_angle3, OMC_elbow_angle1, OMC_elbow_angle2, OMC_elbow_angle3, \
            IMU_shoulder_angle1, IMU_shoulder_angle2, IMU_shoulder_angle3, OMC_shoulder_angle1, OMC_shoulder_angle2, OMC_shoulder_angle3 \
            = calculate_joint_angles(Torso_IMU_df, Torso_OMC_df, Upper_IMU_df, Upper_OMC_df, Forearm_IMU_df, Forearm_OMC_df)

        # Find the joint angle RMSEs
        RMSE_elbow_angle_1, RMSE_elbow_angle_2, RMSE_elbow_angle_3, RMSE_shoulder_angle_1, RMSE_shoulder_angle_2, RMSE_shoulder_angle_3 = \
            find_joint_angle_error(IMU_elbow_angle1, IMU_elbow_angle2, IMU_elbow_angle3, OMC_elbow_angle1, OMC_elbow_angle2, OMC_elbow_angle3, \
            IMU_shoulder_angle1, IMU_shoulder_angle2, IMU_shoulder_angle3, OMC_shoulder_angle1, OMC_shoulder_angle2, OMC_shoulder_angle3)

        elbow_angle_1_RMSEs[i - 1] = RMSE_elbow_angle_1
        elbow_angle_2_RMSEs[i - 1] = RMSE_elbow_angle_2
        elbow_angle_3_RMSEs[i - 1] = RMSE_elbow_angle_3
        shoulder_angle_1_RMSEs[i - 1] = RMSE_shoulder_angle_1
        shoulder_angle_2_RMSEs[i - 1] = RMSE_shoulder_angle_2
        shoulder_angle_3_RMSEs[i - 1] = RMSE_shoulder_angle_3

    elbow_angle_1_RMSD_average = np.mean(elbow_angle_1_RMSEs)
    elbow_angle_1_RMSD_SD = np.std(elbow_angle_1_RMSEs)
    elbow_angle_2_RMSD_average = np.mean(elbow_angle_2_RMSEs)
    elbow_angle_2_RMSD_SD = np.std(elbow_angle_2_RMSEs)
    elbow_angle_3_RMSD_average = np.mean(elbow_angle_3_RMSEs)
    elbow_angle_3_RMSD_SD = np.std(elbow_angle_3_RMSEs)
    shoulder_angle_1_RMSD_average = np.mean(shoulder_angle_1_RMSEs)
    shoulder_angle_1_RMSD_SD = np.std(shoulder_angle_1_RMSEs)
    shoulder_angle_2_RMSD_average = np.mean(shoulder_angle_2_RMSEs)
    shoulder_angle_2_RMSD_SD = np.std(shoulder_angle_2_RMSEs)
    shoulder_angle_3_RMSD_average = np.mean(shoulder_angle_3_RMSEs)
    shoulder_angle_3_RMSD_SD = np.std(shoulder_angle_3_RMSEs)

    logging.info("Movement Type: " + label)
    logging.info("Average ELBOW Results: \n"
                 "Average RMSE - Angle 1: " + str(np.round(elbow_angle_1_RMSD_average, 2)) + " SD: " + str(np.round(elbow_angle_1_RMSD_SD, 2))+ "\n"
                 "Average RMSE - Angle 2: " + str(np.round(elbow_angle_2_RMSD_average, 2)) + " SD: " + str(np.round(elbow_angle_2_RMSD_SD, 2))+ "\n"
                 "Average RMSE - Angle 3: " + str(np.round(elbow_angle_3_RMSD_average, 2)) + " SD: " + str(np.round(elbow_angle_3_RMSD_SD, 2)))
    logging.info("Average SHOULDER Results: \n"
                 "Average RMSE - Angle 1: " + str(np.round(shoulder_angle_1_RMSD_average, 2)) + " SD: " + str(np.round(shoulder_angle_1_RMSD_SD, 2))+ "\n"
                 "Average RMSE - Angle 2: " + str(np.round(shoulder_angle_2_RMSD_average, 2)) + " SD: " + str(np.round(shoulder_angle_2_RMSD_SD, 2))+ "\n"
                 "Average RMSE - Angle 3: " + str(np.round(shoulder_angle_3_RMSD_average, 2)) + " SD: " + str(np.round(shoulder_angle_3_RMSD_SD, 2)))


### APPLY FUNCTIONS DEFINED ABOVE TO FIND JOINT ANGLE RMSE FOR ALL REPS

start_time = 0
end_time = 30
no_reps = 5
combine_reps("FUN")