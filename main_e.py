# This script is used to analyse absolute RMSE Range of Motion results from CON_HO and CON_VE data

import logging
from analysis import *

logging.basicConfig(filename="Results_Abs_RoM.log", level=logging.INFO)
logging.info("RUNNING FILE: " + str(__file__))


def find_RMSD_RoM(file_name, tag, label, plot_peaks):
    # Read in the pre-processed data
    IMU_df, OMC_df = read_abs_pre_processed_data_from_file(file_name)

    # Calculate Euler Angles
    IMU_eul_1, IMU_eul_2, IMU_eul_3 = eulers_from_quats(IMU_df, int_decomp_seq)
    OMC_eul_1, OMC_eul_2, OMC_eul_3 = eulers_from_quats(OMC_df, int_decomp_seq)

    if label == "CON_VE":
        # Analyse rotation around horizontal axis: "Z" which is 2nd in decom seq
        IMU_eul = IMU_eul_2
        OMC_eul = OMC_eul_2
        set_prominence = 10
        set_height = (0, 40)

    elif label == "CON_HO":
        # Analyse rotation around vertical axis: "Y" which is 1st in decom seq
        IMU_eul = -IMU_eul_1
        OMC_eul = -OMC_eul_1
        set_prominence = 10
        set_height = (140, 180)

    # Find the RoMs of IMU and OMC
    IMU_RoMs, OMC_RoMs = find_peaks_and_RoM(IMU_eul, OMC_eul, tag, plot_peaks, set_prominence, set_height)
    # logging.info("IMU ROMS: "), logging.info(IMU_RoMs)
    # logging.info("OMC ROMS: "), logging.info(OMC_RoMs)

    # Calculate the difference between the IMU and OMC range of motion
    RoM_error = np.array([abs(IMU_RoMs - OMC_RoMs)])

    # Calculate RMSE of the four RoM Errors in this rep
    RMSD_RoM = (np.sum(np.square(RoM_error)) / len(RoM_error)) ** 0.5

    return RMSD_RoM


def combine_reps_RoM(label, plot_peaks):
    # Initiate arrays to hold a single value for each rep
    RoM_RMSDs = np.zeros((no_reps))

    for i in range(1, no_reps + 1):
        file_name = label + "_R" + str(i) + ".csv"
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


