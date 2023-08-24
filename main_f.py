# This script is used to analyse inter-IMU RMSE Range of Motion results from CON_HO and CON_VE data

import logging
from analysis import *

logging.basicConfig(filename="Results_Inter_RoM.log", level=logging.INFO)
logging.info("RUNNING FILE: " + str(__file__))



def find_RMSD_RoM(file_name, tag, label, plot_peaks):

    # Read in the pre-processed data
    IMU1_df, IMU2_df, IMU3_df, IMU4_df = read_inter_pre_processed_data_from_file(file_name)

    # Calculate Euler Angles
    IMU1_eul_1, IMU1_eul_2, IMU1_eul_3 = eulers_from_quats(IMU1_df, int_decomp_seq)
    IMU2_eul_1, IMU2_eul_2, IMU2_eul_3 = eulers_from_quats(IMU2_df, int_decomp_seq)
    IMU3_eul_1, IMU3_eul_2, IMU3_eul_3 = eulers_from_quats(IMU3_df, int_decomp_seq)
    IMU4_eul_1, IMU4_eul_2, IMU4_eul_3 = eulers_from_quats(IMU4_df, int_decomp_seq)

    if label == "CON_VE":
        # Analyse rotation around horizontal axis: "Z" which is 2nd in decom seq
        IMU1_eul = IMU1_eul_2
        IMU2_eul = IMU2_eul_2
        IMU3_eul = IMU3_eul_2
        IMU4_eul = IMU4_eul_2
        set_prominence = 10
        set_height = (0, 40)

    elif label == "CON_HO":
        # Analyse rotation around vertical axis: "Y" which is 1st in decom seq
        IMU1_eul = -IMU1_eul_1
        IMU2_eul = -IMU2_eul_1
        IMU3_eul = -IMU3_eul_1
        IMU4_eul = -IMU4_eul_1
        set_prominence = 10
        set_height = (100, 140)

    # Find the RoMs of IMU and OMC
    IMU1_RoMs, IMU2_RoMs, IMU3_RoMs, IMU4_RoMs = find_peaks_and_RoM_four_IMUs(IMU1_eul, IMU2_eul, IMU3_eul, IMU4_eul, tag, plot_peaks, set_prominence, set_height)

    # Calculate the RMSD between the four IMUs for the RoM value at each peak
    peak1_RoM = np.array([IMU1_RoMs[0,0], IMU2_RoMs[0,0], IMU3_RoMs[0,0], IMU4_RoMs[0,0]])
    peak2_RoM = np.array([IMU1_RoMs[0,1], IMU2_RoMs[0,1], IMU3_RoMs[0,1], IMU4_RoMs[0,1]])
    peak3_RoM = np.array([IMU1_RoMs[0,2], IMU2_RoMs[0,2], IMU3_RoMs[0,2], IMU4_RoMs[0,2]])
    peak4_RoM = np.array([IMU1_RoMs[0,3], IMU2_RoMs[0,3], IMU3_RoMs[0,3], IMU4_RoMs[0,3]])
    peak1_RoM_RMSD = (np.sum(np.square(peak1_RoM - np.mean(peak1_RoM)))/len(peak1_RoM))**0.5
    peak2_RoM_RMSD = (np.sum(np.square(peak2_RoM - np.mean(peak2_RoM)))/len(peak2_RoM))**0.5
    peak3_RoM_RMSD = (np.sum(np.square(peak3_RoM - np.mean(peak3_RoM)))/len(peak3_RoM))**0.5
    peak4_RoM_RMSD = (np.sum(np.square(peak4_RoM - np.mean(peak4_RoM)))/len(peak4_RoM))**0.5

    # Combine the peak RoM values into one array so that RMSD results can be averaged over all reps
    peak_RMSDs = np.array([peak1_RoM_RMSD, peak2_RoM_RMSD, peak3_RoM_RMSD, peak4_RoM_RMSD])
    average_RoM_RMSD = np.mean(peak_RMSDs)

    return average_RoM_RMSD



def combine_reps_RoM(label, plot_peaks):

    # Initiate arrays to hold a single value for each rep
    RoM_RMSDs = np.zeros((no_reps))

    for i in range(1, no_reps + 1):

        file_name = "Inter_" + label + "_R" + str(i) + ".csv"
        tag = label + "_R" + str(i)

        # Find RoM from euler angles
        RMSD_RoM = find_RMSD_RoM(file_name, tag, label, plot_peaks=plot_peaks)

        # Add the value from each rep to the arrays
        RoM_RMSDs[i - 1] = RMSD_RoM

    RoM_RMSD_average = np.mean(RoM_RMSDs)
    RoM_RMSD_SD = np.std(RoM_RMSDs)

    logging.info("Movement Type: " + label)
    logging.info("\n" +
                 "Average RoM RMSD: " + str(np.around(RoM_RMSD_average, 2)) + "\n" +
                 "Average RoM SD: " + str(np.around(RoM_RMSD_SD, 2)))



### APPLY FUNCTIONS DEFINED ABOVE TO FIND AVERAGE RMSE FOR ALL REPS

start_time = 0
end_time = 32
no_reps = 5
int_decomp_seq = "YZX"  # Intrinsic decomposition seq - used for plotting euler angles
logging.info("Int Decomp Sequence: " + int_decomp_seq)
plot_peaks = False

combine_reps_RoM("CON_HO", plot_peaks=plot_peaks)
combine_reps_RoM("CON_VE", plot_peaks=plot_peaks)


