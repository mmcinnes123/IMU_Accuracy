# This script is used to analyse absolute RMSE results from CON_MP and CON_SP and FUN data

import logging
from analysis import *


logging.basicConfig(filename="Results_Abs.log", level=logging.INFO)
logging.info("RUNNING FILE: " + str(__file__))

# SETTINGS
sample_rate = 100
ext_decomp_seq = "yxz"  # Extrinsic decomposition seq - used for calculating angular distance

### ABSOLUTE ACCURACY TESTS

def trim_data(input_file, label, start_time, end_time):

    IMU_df, OMC_df = read_abs_pre_processed_data_from_file(input_file)

    # Trim the data based on start and end time
    IMU_df = trim_df(IMU_df, start_time, end_time, sample_rate)
    OMC_df = trim_df(OMC_df, start_time, end_time, sample_rate)

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
        IMU_df = cut_four_sections(IMU_df, s1, e1, s2, e2, s3, e3, s4, e4, sample_rate)
        OMC_df = cut_four_sections(OMC_df, s1, e1, s2, e2, s3, e3, s4, e4, sample_rate)

    return IMU_df, OMC_df


def find_abs_error_CON(IMU_df, OMC_df):

    # Find the 3-angle quaternion-based orientation difference between an IMU and OMC LCF.
    RMSD_angle_1, RMSD_angle_2, RMSD_angle_3 = find_quat_diff(IMU_df, OMC_df, ext_decomp_seq)[3:6]

    # Find the single-angle quaternion-based orientation difference between an IMU and OMC LCF.
    ang_dist, RMSD_ang_dist = find_ang_dist(IMU_df, OMC_df)

    # Find correlation of IMU error with time
    corr_coeff = find_corr_coef(ang_dist)

    return RMSD_angle_1, RMSD_angle_2, RMSD_angle_3, RMSD_ang_dist, ang_dist, corr_coeff


def combine_reps_CON_MP_SP_FUN(label, no_reps, start_time, end_time):

    # Initiate arrays to hold a single value for each rep
    angle_1_RMSDs = np.zeros((no_reps))
    angle_2_RMSDs = np.zeros((no_reps))
    angle_3_RMSDs = np.zeros((no_reps))
    ang_dist_RMSDs = np.zeros((no_reps))
    corr_coeffs = np.zeros((no_reps))

    for i in range(1, no_reps+1):

        file_name = label + "_R" + str(i) + ".csv"

        # Trim the data down to sections of interest
        IMU_df, OMC_df = trim_data(file_name, label, start_time, end_time)

        # Calculate the time-series RMSD and correlation coefficient in every rep
        RMSD_angle_1, RMSD_angle_2, RMSD_angle_3, RMSD_ang_dist, ang_dist, corr_coeff = find_abs_error_CON(IMU_df,
                                                                                                           OMC_df)
        # Add the value from each rep to the arrays
        angle_1_RMSDs[i-1] = RMSD_angle_1
        angle_2_RMSDs[i-1] = RMSD_angle_2
        angle_3_RMSDs[i-1] = RMSD_angle_3
        ang_dist_RMSDs[i-1] = RMSD_ang_dist
        corr_coeffs[i-1] = corr_coeff

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

    # Calculate the average correlation coefficient across all reps
    corr_coeff_average = np.mean(corr_coeffs)
    corr_coeff_SD = np.std(corr_coeffs)

    # Log the result
    logging.info("Movement Type: " + label)
    logging.info("Average Results: \n"
                 "(Decomp seq: " + ext_decomp_seq + ")\n"
                 "Average RMSE - Angle 1: " + str(np.around(angle_1_RMSD_average, 2)) + " SD: " + str(np.around(angle_1_RMSD_SD, 2)) + "\n"
                 "Average RMSE - Angle 2: " + str(np.around(angle_2_RMSD_average, 2)) + " SD: " + str(np.around(angle_2_RMSD_SD, 2)) + "\n"
                 "Average RMSE - Angle 3: " + str(np.around(angle_3_RMSD_average, 2)) + " SD: " + str(np.around(angle_3_RMSD_SD, 2)) + "\n"
                 "Average RMSE - Angles 2 and 3: " + str(np.around(angle_2_and_3_RMSD_average, 2)) + " SD: " + str(np.around(angle_2_and_3_RMSD_SD, 2)) + "\n"
                 "Average RMSE - Angular Distance: " + str(np.around(ang_dist_RMSD_average, 2)) + " SD: " + str(np.around(ang_dist_RMSD_SD, 2)) + "\n"
                 "Average Correlation Coefficient (AngDist vs Time): " + str(np.around(corr_coeff_average, 2)) + " SD:" + str(np.around(corr_coeff_SD, 2)))



### APPLY FUNCTIONS DEFINED ABOVE TO FIND AVERAGE RMSE FOR ALL REPS

combine_reps_CON_MP_SP_FUN(label="CON_MP", no_reps=5, start_time=0, end_time=30)
combine_reps_CON_MP_SP_FUN(label="CON_SP", no_reps=3, start_time=0, end_time=120)

combine_reps_CON_MP_SP_FUN(label="Torso_FUN", no_reps=5, start_time=0, end_time=30)
combine_reps_CON_MP_SP_FUN(label="Upper_FUN", no_reps=5, start_time=0, end_time=30)
combine_reps_CON_MP_SP_FUN(label="Forearm_FUN", no_reps=5, start_time=0, end_time=30)
