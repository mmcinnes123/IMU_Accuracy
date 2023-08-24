# This script is used to analyse inter-IMU RMSE results from CON_MP and CON_SP data

import logging
from analysis import *


logging.basicConfig(filename="Results_Inter.log", level=logging.INFO)
logging.info("RUNNING FILE: " + str(__file__))

# SETTINGS
sample_rate = 100
ext_decomp_seq = "yxz"  # Extrinsic decomposition seq - used for calculating angular distance

### ABSOLUTE ACCURACY TESTS

def trim_data_inter(input_file, label, start_time, end_time):

    IMU1_df, IMU2_df, IMU3_df, IMU4_df = read_inter_pre_processed_data_from_file(input_file)

    # Trim the data based on start and end time
    IMU1_df = trim_df(IMU1_df, start_time, end_time, sample_rate)
    IMU2_df = trim_df(IMU2_df, start_time, end_time, sample_rate)
    IMU3_df = trim_df(IMU3_df, start_time, end_time, sample_rate)
    IMU4_df = trim_df(IMU4_df, start_time, end_time, sample_rate)

    if label == "CON_SP":
        # Trim the data so that only the static sections are analysed
        s1 = 12.5
        e1 = 27.5
        s2 = 42.5
        e2 = 57.5
        s3 = 72.5
        e3 = 87.5
        s4 = 102.5
        e4 = 117.5
        IMU1_df = cut_four_sections(IMU1_df, s1, e1, s2, e2, s3, e3, s4, e4, sample_rate)
        IMU2_df = cut_four_sections(IMU2_df, s1, e1, s2, e2, s3, e3, s4, e4, sample_rate)
        IMU3_df = cut_four_sections(IMU3_df, s1, e1, s2, e2, s3, e3, s4, e4, sample_rate)
        IMU4_df = cut_four_sections(IMU4_df, s1, e1, s2, e2, s3, e3, s4, e4, sample_rate)

    return IMU1_df, IMU2_df, IMU3_df, IMU4_df


def find_inter_error_CON(IMU1_df, IMU2_df, IMU3_df, IMU4_df):

    # For every time sample, calculate an average quaternion which represents the midpoint between all 4 IMUs.
    comparison_quats = av_quat_from_four_IMUs(IMU1_df, IMU2_df, IMU3_df, IMU4_df)

    # Find the single-angle quaternion-based orientation difference between each IMU and the comparison quats
    IMU1_ang_dist, IMU1_ang_dist_RMSD = find_ang_dist(IMU1_df, comparison_quats)
    IMU2_ang_dist, IMU2_ang_dist_RMSD = find_ang_dist(IMU2_df, comparison_quats)
    IMU3_ang_dist, IMU3_ang_dist_RMSD = find_ang_dist(IMU3_df, comparison_quats)
    IMU4_ang_dist, IMU4_ang_dist_RMSD = find_ang_dist(IMU4_df, comparison_quats)

    # Find the 3-angle quaternion difference between each IMU and the comparison quats
    IMU1_angle_1,  IMU1_angle_2, IMU1_angle_3 = find_quat_diff(IMU1_df, comparison_quats, ext_decomp_seq)[0:3]
    IMU2_angle_1,  IMU2_angle_2, IMU2_angle_3 = find_quat_diff(IMU2_df, comparison_quats, ext_decomp_seq)[0:3]
    IMU3_angle_1,  IMU3_angle_2, IMU3_angle_3 = find_quat_diff(IMU3_df, comparison_quats, ext_decomp_seq)[0:3]
    IMU4_angle_1,  IMU4_angle_2, IMU4_angle_3 = find_quat_diff(IMU4_df, comparison_quats, ext_decomp_seq)[0:3]

    # Calculate the RMSD angular error at each time sample (where error is angular distance between IMU and averaged orientation
    RMSD_ang_dist = find_RMSD_four_IMUS(IMU1_ang_dist, IMU2_ang_dist, IMU3_ang_dist, IMU4_ang_dist)
    RMSD_angle_1 = find_RMSD_four_IMUS(IMU1_angle_1, IMU2_angle_1, IMU3_angle_1, IMU4_angle_1)
    RMSD_angle_2 = find_RMSD_four_IMUS(IMU1_angle_2, IMU2_angle_2, IMU3_angle_2, IMU4_angle_2)
    RMSD_angle_3 = find_RMSD_four_IMUS(IMU1_angle_3, IMU2_angle_3, IMU3_angle_3, IMU4_angle_3)

    # Calculate average RMSD for each rep
    average_RMSD_ang_dist = np.mean(RMSD_ang_dist)
    average_RMSD_angle_1 = np.mean(RMSD_angle_1)
    average_RMSD_angle_2 = np.mean(RMSD_angle_2)
    average_RMSD_angle_3 = np.mean(RMSD_angle_3)

    return average_RMSD_ang_dist, average_RMSD_angle_1, average_RMSD_angle_2, average_RMSD_angle_3


def combine_reps_CON_MP_or_SP(label, no_reps, start_time, end_time):

    # Initiate arrays to hold a single value for each rep
    angle_1_RMSDs = np.zeros((no_reps))
    angle_2_RMSDs = np.zeros((no_reps))
    angle_3_RMSDs = np.zeros((no_reps))
    ang_dist_RMSDs = np.zeros((no_reps))
    corr_coeffs = np.zeros((no_reps))

    for i in range(1, no_reps+1):

        file_name = "Inter_" + label + "_R" + str(i) + ".csv"

        # Trim the data down to sections of interest
        IMU1_df, IMU2_df, IMU3_df, IMU4_df = trim_data_inter(file_name, label, start_time, end_time)

        # Calculate the time-series RMSD and correlation coefficient in every rep
        rep_i_RMSD_ang_dist, rep_i_RMSD_angle_1, rep_i_RMSD_angle_2, rep_i_RMSD_angle_3 = find_inter_error_CON(IMU1_df, IMU2_df, IMU3_df, IMU4_df)

        # Add the average value from each rep to the arrays
        angle_1_RMSDs[i-1] = rep_i_RMSD_angle_1
        angle_2_RMSDs[i-1] = rep_i_RMSD_angle_2
        angle_3_RMSDs[i-1] = rep_i_RMSD_angle_3
        ang_dist_RMSDs[i-1] = rep_i_RMSD_ang_dist

    # Calculate the average RMSD across the five reps, and the standard deviation in that RMSD
    angle_1_RMSD_average = np.mean(angle_1_RMSDs)
    angle_1_RMSD_SD = np.std(angle_1_RMSDs)
    angle_2_RMSD_average = np.mean(angle_2_RMSDs)
    angle_2_RMSD_SD = np.std(angle_2_RMSDs)
    angle_3_RMSD_average = np.mean(angle_3_RMSDs)
    angle_3_RMSD_SD = np.std(angle_3_RMSDs)
    ang_dist_RMSD_average = np.mean(ang_dist_RMSDs)
    ang_dist_RMSD_SD = np.std(ang_dist_RMSDs)

    # Take an average of the vertical planes (i.e. euler angle 2 and 3 (in the YXZ decomposition sequence))
    angle_2_and_3_RMSD_average = np.mean(np.concatenate([angle_2_RMSDs, angle_3_RMSDs]))
    angle_2_and_3_RMSD_SD = np.std(np.concatenate([angle_2_RMSDs, angle_3_RMSDs]))


    # Log the result
    logging.info("Movement Type: " + label)
    logging.info("Average Results: \n"
                 "(Decomp seq: " + ext_decomp_seq + ")\n"
                 "Average RMSE - Angle 1: " + str(np.around(angle_1_RMSD_average, 2)) + " SD: " + str(np.around(angle_1_RMSD_SD, 2)) + "\n"
                 "Average RMSE - Angle 2: " + str(np.around(angle_2_RMSD_average, 2)) + " SD: " + str(np.around(angle_2_RMSD_SD, 2)) + "\n"
                 "Average RMSE - Angle 3: " + str(np.around(angle_3_RMSD_average, 2)) + " SD: " + str(np.around(angle_3_RMSD_SD, 2)) + "\n"
                 "Average RMSE - Angles 2 and 3: " + str(np.around(angle_2_and_3_RMSD_average, 2)) + " SD: " + str(np.around(angle_2_and_3_RMSD_SD, 2)) + "\n"
                 "Average RMSE - Angular Distance: " + str(np.around(ang_dist_RMSD_average, 2)) + " SD: " + str(np.around(ang_dist_RMSD_SD, 2)) + "\n")


combine_reps_CON_MP_or_SP(label="CON_MP", no_reps=5, start_time=0, end_time=30)
combine_reps_CON_MP_or_SP(label="CON_SP", no_reps=3, start_time=0, end_time=120)
