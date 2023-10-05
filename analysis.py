### Functions used to analyse the data

import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from quat_functions import *
import scipy.signal


# Read pre-processed data file specific to absolute error
def read_abs_pre_processed_data_from_file(input_file):
    with open("ProcessedData/" + input_file, 'r') as file:
        df = pd.read_csv(file, header=0, sep=",")
    # Make seperate data frames
    IMU_df = df.filter(["IMU_Q0", "IMU_Q1", "IMU_Q2", "IMU_Q3"], axis=1)
    OMC_df = df.filter(["OMC_Q0", "OMC_Q1", "OMC_Q2", "OMC_Q3"], axis=1)
    return IMU_df, OMC_df


# Read pre-processed data file specific to inter-IMU agreement
def read_inter_pre_processed_data_from_file(input_file):
    with open("ProcessedData/" + input_file, 'r') as file:
        df = pd.read_csv(file, header=0, sep=",")
    # Make seperate data frames
    IMU1_df = df.filter(["IMU1_Q0", "IMU1_Q1", "IMU1_Q2", "IMU1_Q3"], axis=1)
    IMU2_df = df.filter(["IMU2_Q0", "IMU2_Q1", "IMU2_Q2", "IMU2_Q3"], axis=1)
    IMU3_df = df.filter(["IMU3_Q0", "IMU3_Q1", "IMU3_Q2", "IMU3_Q3"], axis=1)
    IMU4_df = df.filter(["IMU4_Q0", "IMU4_Q1", "IMU4_Q2", "IMU4_Q3"], axis=1)
    return IMU1_df, IMU2_df, IMU3_df, IMU4_df


# Fidn the average quaternion from an array of four IMUs
def av_quat_from_four_IMUs(IMU1_df, IMU2_df, IMU3_df, IMU4_df):
    N = len(IMU1_df)
    q_avg = np.empty((N,4))
    for row in range(N):
        q1_i = IMU1_df.values[row].reshape((1,4))
        q2_i = IMU2_df.values[row].reshape((1,4))
        q3_i = IMU3_df.values[row].reshape((1,4))
        q4_i = IMU4_df.values[row].reshape((1,4))
        q_arr_i = np.concatenate((q1_i, q2_i, q3_i, q4_i), axis=0)
        q_avg[row] = average_quaternions(q_arr_i)
    q_avg_df = pd.DataFrame(q_avg)
    return q_avg_df


# Find RMSD from error (angular distance) values between four IMUs
def find_RMSD_four_IMUS(IMU1_ang_dist, IMU2_ang_dist, IMU3_ang_dist, IMU4_ang_dist):
    N = len(IMU1_ang_dist)
    RMSD = np.empty((N))
    for i in range(N):
        RMSD[i] = (np.mean(np.square([IMU1_ang_dist[i], IMU2_ang_dist[i], IMU3_ang_dist[i], IMU4_ang_dist[i]]))) ** 0.5
    return RMSD


# Find quaternion-based difference in orientation between one IMU and a cluster
def find_quat_diff(IMU_df, Clust_df, decomp_seq):
    # For every time sample in the data frame, calculate the rotational difference between the IMU and the cluster, then decompose into Euler angles
    N = len(IMU_df)
    diff_angle_1 = np.zeros((N))
    diff_angle_2 = np.zeros((N))
    diff_angle_3 = np.zeros((N))
    for row in range(N):
        quat_IMU = IMU_df.values[row]
        quat_Clust = Clust_df.values[row]
        q_diff = quat_mul(quat_IMU, quat_conj(quat_Clust))
        q_diff_R = R.from_quat([q_diff[1], q_diff[2], q_diff[3], q_diff[0]])
        eul_diff = q_diff_R.as_euler(decomp_seq, degrees=True)
        diff_angle_1[row] = eul_diff[0]
        diff_angle_2[row] = eul_diff[1]
        diff_angle_3[row] = eul_diff[2]

    ### Calculate averages
    RMSD_angle_1 = (sum(np.square(diff_angle_1))/len(diff_angle_1))**0.5
    RMSD_angle_2 = (sum(np.square(diff_angle_2))/len(diff_angle_2))**0.5
    RMSD_angle_3 = (sum(np.square(diff_angle_3))/len(diff_angle_3))**0.5

    return diff_angle_1, diff_angle_2, diff_angle_3, RMSD_angle_1, RMSD_angle_2, RMSD_angle_3


# Find angular distance between two (data frames) of data-paired quaternions
# (Note: cos function is sensitive to inputs which can be just over 1. Warning is issued and value is rounded to 1.)
def find_ang_dist(IMU_df, Clust_df):
    angle = np.zeros((len(IMU_df)))
    for row in range(len(IMU_df)):
        dot_prod = quat_dot_prod(IMU_df.values[row], Clust_df.values[row])
        if abs(dot_prod) > 1:
            print("Warning: input to cos function outside range (-1, 1)")
            print("Value:" + str(dot_prod))
            dot_prod = round(dot_prod)
            print("Value used instead:" + str(dot_prod))
        angle[row] = 2 * np.arccos(abs(dot_prod)) * 180 / np.pi
    RMSD_angle = (sum(np.square(angle))/len(angle))**0.5
    return angle, RMSD_angle


# Find correlation between error and time
def find_corr_coef(single_angle_diff):
    time = np.array(list(np.arange(0, len(single_angle_diff) / 100, 1 / 100)))
    corr_coeff = np.corrcoef(single_angle_diff, time)[0,1]
    return corr_coeff


### EULER ANGLES

# Calculate Euler angles from quaternions
def eulers_from_quats(quat_df, decomp_seq):
    quat_df.dropna()
    N = len(quat_df)
    Eul_angle1 = np.zeros((N))
    Eul_angle2 = np.zeros((N))
    Eul_angle3 = np.zeros((N))
    for row in range(N):
        quat = quat_df.values[row]
        quat_R = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        eul = quat_R.as_euler(decomp_seq, degrees=True)
        Eul_angle1[row] = eul[0]
        Eul_angle2[row] = eul[1]
        Eul_angle3[row] = eul[2]

    return Eul_angle1, Eul_angle2, Eul_angle3


# Adjusts shoulder angles by 180 degrees so that convention matches Pearl paper
def adjust_angles_by_180(angle_arr):
    for i in range(len(angle_arr)):
        if angle_arr[i] < 0:
            angle_arr[i] = angle_arr[i] + 180
        else:
            angle_arr[i] = angle_arr[i] - 180
    return angle_arr


# Rotatoes thorax coordinate frame by 90 degrees to align with ISB definition
def rotate_thorax(quat_df):
    N = len(quat_df)
    transformed_quats = np.zeros((N, 4))
    rot_matrix = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    rot_matrix_asquat = R.from_matrix(rot_matrix).as_quat()
    rot_quat = [rot_matrix_asquat[3], rot_matrix_asquat[0], rot_matrix_asquat[1], rot_matrix_asquat[2]]
    for row in range(N):
        quat_i = np.array([quat_df.values[row,0], quat_df.values[row,1], quat_df.values[row,2], quat_df.values[row,3]])
        transformed_quats[row] = quat_mul(quat_i, rot_quat)
    transformed_quats_df = pd.DataFrame(transformed_quats)
    return transformed_quats_df


# Calculates the joint rotation quaternion, used to find euler angles
def find_joint_rot_quat(q_df_prox, q_df_dist):
    N = len(q_df_dist)
    q_joint_arr = np.zeros((N, 4))
    for row in range(N):
        q_prox_i = np.array([q_df_prox.values[row, 0], q_df_prox.values[row,1], q_df_prox.values[row,2], q_df_prox.values[row,3]])
        q_dist_i = np.array([q_df_dist.values[row, 0], q_df_dist.values[row,1], q_df_dist.values[row,2], q_df_dist.values[row,3]])
        q_joint_arr[row] = quat_mul(quat_conj(q_prox_i), q_dist_i)
    q_joint_df = pd.DataFrame(q_joint_arr)
    return q_joint_df


# Find error based on euler angles
def find_euler_error(IMU_angles, OMC_angles):
    error = abs(IMU_angles - OMC_angles)
    # If error is over 100 degrees, remove from the array
    error = error[error < 300]
    RMSD = (sum(np.square(error)) / len(IMU_angles)) ** 0.5
    return RMSD


# Find peaks of a signal and calculate 'range of motion' from peak value relative to initial angle
def find_peaks_and_RoM(IMU_eul, OMC_eul, tag, plot_peaks, set_prominence, set_height):

    # Find the value of the relevant euler angle at each peak in the data
    IMU_peaks, IMU_properties = scipy.signal.find_peaks(IMU_eul, prominence=set_prominence, height=set_height)
    OMC_peaks, OMC_properties = scipy.signal.find_peaks(OMC_eul, prominence=set_prominence, height=set_height)

    if plot_peaks == True:
        plt.figure(1)
        plt.plot(OMC_peaks, OMC_eul[OMC_peaks], "ob")
        plt.plot(OMC_eul, c='blue')
        plt.plot(IMU_peaks, IMU_eul[IMU_peaks], "xr")
        plt.plot(IMU_eul, c='red')
        plt.rcParams.update({'figure.figsize': (20, 16), 'figure.dpi': 200})
        plt.title("Identifying peaks")
        plt.xlabel('Time sample')
        plt.ylabel('Angle [deg]')
        plt.legend(["OMC Peak", "OMC Euler Angle", "IMU Peak", "IMU Euler Angle"], loc="lower right", markerscale=1)
        plt.grid(axis="x", which="both")
        plt.savefig("Peaks_" + tag + ".png")
        plt.clf()

    # Only use first 4 peaks
    IMU_peaks = IMU_peaks[0:4]
    OMC_peaks = OMC_peaks[0:4]

    IMU_RoMs = np.array([IMU_eul[IMU_peaks] - [IMU_eul[0], IMU_eul[0], IMU_eul[0], IMU_eul[0]]])
    OMC_RoMs = np.array([OMC_eul[OMC_peaks] - [OMC_eul[0], OMC_eul[0], OMC_eul[0], OMC_eul[0]]])

    return IMU_RoMs, OMC_RoMs


# Find peaks and range of motion for four imus simultaneously.
def find_peaks_and_RoM_four_IMUs(IMU1_eul, IMU2_eul, IMU3_eul, IMU4_eul, tag, plot_peaks, set_prominence, set_height):

    # Find the value of the relevant euler angle at each peak in the data
    IMU1_peaks, IMU_properties = scipy.signal.find_peaks(IMU1_eul, prominence=set_prominence, height=set_height)
    IMU2_peaks, IMU_properties = scipy.signal.find_peaks(IMU2_eul, prominence=set_prominence, height=set_height)
    IMU3_peaks, IMU_properties = scipy.signal.find_peaks(IMU3_eul, prominence=set_prominence, height=set_height)
    IMU4_peaks, IMU_properties = scipy.signal.find_peaks(IMU4_eul, prominence=set_prominence, height=set_height)

    if plot_peaks == True:
        time = np.array(list(np.arange(0, len(IMU1_eul) / 100, 1 / 100)))
        x = time
        plt.figure(1)
        plt.plot(x, IMU1_eul, c='red')
        plt.plot(x, IMU2_eul, c='darkred')
        plt.plot(x, IMU3_eul, c='firebrick')
        plt.plot(x, IMU4_eul, c='salmon')
        plt.plot(IMU1_peaks / 100, IMU1_eul[IMU1_peaks], "x", c='red')
        plt.plot(IMU2_peaks / 100, IMU2_eul[IMU2_peaks], "x", c='darkred')
        plt.plot(IMU3_peaks / 100, IMU3_eul[IMU3_peaks], "x", c='firebrick')
        plt.plot(IMU4_peaks / 100, IMU4_eul[IMU4_peaks], "x", c='salmon')
        plt.rcParams.update({'figure.figsize': (20, 16), 'figure.dpi': 200})
        plt.title("Range of Motion - Vertical Plane")
        plt.xlabel('Time [s]')
        plt.ylabel('Euler Angle (horizontal axis) [deg]')
        plt.legend(["IMU 1", "IMU 2", "IMU 3", "IMU 4", "IMU Peaks"], loc="lower right", markerscale=1)
        plt.grid(axis="x", which="both")
        plt.grid(axis="y", which="both")
        plt.savefig("Peaks_" + tag + ".png")
        plt.clf()

    # Only use first 4 peaks
    IMU1_peaks = IMU1_peaks[0:4]
    IMU2_peaks = IMU2_peaks[0:4]
    IMU3_peaks = IMU3_peaks[0:4]
    IMU4_peaks = IMU4_peaks[0:4]

    IMU1_RoMs = np.array([IMU1_eul[IMU1_peaks] - [IMU1_eul[0], IMU1_eul[0], IMU1_eul[0], IMU1_eul[0]]])
    IMU2_RoMs = np.array([IMU2_eul[IMU2_peaks] - [IMU2_eul[0], IMU2_eul[0], IMU2_eul[0], IMU2_eul[0]]])
    IMU3_RoMs = np.array([IMU3_eul[IMU3_peaks] - [IMU3_eul[0], IMU3_eul[0], IMU3_eul[0], IMU3_eul[0]]])
    IMU4_RoMs = np.array([IMU4_eul[IMU4_peaks] - [IMU4_eul[0], IMU4_eul[0], IMU4_eul[0], IMU4_eul[0]]])

    return IMU1_RoMs, IMU2_RoMs, IMU3_RoMs, IMU4_RoMs


### PLOTTING FUNCTIONS

def plot_IMU_vs_Clus_eulers_elbow(IMU_eul_1, IMU_eul_2, IMU_eul_3, Clust_eul_1, Clust_eul_2, Clust_eul_3, decomp_seq, tag, sample_rate):
    plt.rcParams.update({'figure.figsize': (11,6), 'figure.dpi': 300})
    figure = plt.figure()
    no_secs = len(IMU_eul_1) / 100
    time = np.array(list(np.arange(0, no_secs, 0.01)))
    font_size_title = 24
    font_size_axes = 20
    font_size_legend = 16
    font_size_ticks = 18
    # time = np.array(list(np.arange(0, len(IMU_eul_1) / 6000, 5 / len(IMU_eul_1))))
    plt.plot(time, Clust_eul_1, c='darkred', linewidth=4)
    plt.plot(time, IMU_eul_1, c='lightcoral', linewidth=4)
    plt.plot(time, Clust_eul_2, c='darkslategrey', linewidth=4)
    plt.plot(time, IMU_eul_2, c='lightseagreen', linewidth=4)
    plt.plot(time, Clust_eul_3, c='orangered', linewidth=4)
    plt.plot(time, IMU_eul_3, c='lightsalmon', linewidth=4)
    figure.suptitle("Figure 3: Elbow joint angles - IMU vs OMC", fontsize=font_size_title, weight='bold')
    plt.ylabel('Joint Angle (' + chr(176) + ')', fontdict={'fontsize': font_size_axes})
    plt.xlabel('Time (s)', labelpad=6.0, fontdict={'fontsize': font_size_axes})
    plt.ylim((-100, 200))
    plt.xlim((0,no_secs))
    plt.xticks(range(0, int(no_secs)+5, 5), fontsize=font_size_ticks)
    plt.yticks(range(-150, 250, 50), fontsize=font_size_ticks)
    plt.legend(["OMC Flex/Ext", "IMU Flex/Ext", "OMC Ab/Add", "IMU Ab/Add", "OMC Pro/Sup", "IMU Pro/Sup"], loc="lower center", markerscale=3, ncol=3, fontsize=font_size_legend, facecolor='white', framealpha=1)
    # plt.subplots_adjust(bottom=0.1, top=0.85, right=0.95, left=0.05)
    plt.savefig("IMUvsOMC_Eulers_" + tag + "_" + decomp_seq + ".png")
    plt.clf()


def plot_IMU_vs_Clus_eulers_shoulder(IMU_eul_1, IMU_eul_2, IMU_eul_3, Clust_eul_1, Clust_eul_2, Clust_eul_3, decomp_seq, tag, sample_rate):
    plt.rcParams.update({'figure.figsize': (11,6), 'figure.dpi': 300})
    figure = plt.figure()
    no_secs = len(IMU_eul_1) / 100
    time = np.array(list(np.arange(0, no_secs, 0.01)))
    font_size_title = 24
    font_size_axes = 20
    font_size_legend = 16
    font_size_ticks = 18
    # time = np.array(list(np.arange(0, len(IMU_eul_1) / 6000, 5 / len(IMU_eul_1))))
    plt.plot(time, Clust_eul_1, c='darkred', linewidth=4)
    plt.plot(time, IMU_eul_1, c='lightcoral', linewidth=4)
    plt.plot(time, Clust_eul_2, c='darkslategrey', linewidth=4)
    plt.plot(time, IMU_eul_2, c='lightseagreen', linewidth=4)
    plt.plot(time, Clust_eul_3, c='orangered', linewidth=4)
    plt.plot(time, IMU_eul_3, c='lightsalmon', linewidth=4)
    figure.suptitle("Figure 3: Shoulder joint angles - IMU vs OMC", fontsize=font_size_title, weight='bold')
    plt.ylabel('Joint Angle (' + chr(176) + ')', fontdict={'fontsize': font_size_axes})
    plt.xlabel('Time (s)', labelpad=6.0, fontdict={'fontsize': font_size_axes})
    plt.ylim((-120, 120))
    plt.xlim((0,no_secs))
    plt.xticks(range(0, int(no_secs)+5, 5), fontsize=font_size_ticks)
    plt.yticks(range(-100, 150, 50), fontsize=font_size_ticks)
    plt.legend(["OMC Y1", "IMU Y1", "OMC X", "IMU X", "OMC Y2", "IMU Y2"], loc="lower center", markerscale=3, ncol=3, fontsize=font_size_legend, facecolor='white', framealpha=1)
    # plt.subplots_adjust(bottom=0.1, top=0.85, right=0.95, left=0.05)
    plt.savefig("IMUvsOMC_Eulers_" + tag + "_" + decomp_seq + ".png")
    plt.clf()


def plot_all_errors_vs_time(T_diff_R1, T_diff_R2, T_diff_R3, T_diff_R4, T_diff_R5,
                       U_diff_R1, U_diff_R2, U_diff_R3, U_diff_R4, U_diff_R5,
                       F_diff_R1, F_diff_R2, F_diff_R3, F_diff_R4, F_diff_R5,
                       average_RMSD_T, average_RMSD_U, average_RMSD_F,
                       average_SD_T, average_SD_U, average_SD_F):


    # Define the parameters to be used in the plot
    no_mins = len(T_diff_R1) / 6000
    time = np.array(list(np.arange(0, no_mins, no_mins / len(T_diff_R1))))
    forearm_colour = "coral"
    upper_colour = "teal"
    thorax_colour = "firebrick"
    grid_colour = "whitesmoke"
    torso_RMSE = average_RMSD_T
    torso_SD = average_SD_T
    upper_RMSE = average_RMSD_U
    upper_SD = average_SD_U
    forearm_RMSE = average_RMSD_F
    forearm_SD = average_SD_F
    y_range_max = 40
    font_size_title = 24
    font_size_axes = 20
    font_size_legend = 18
    font_size_ticks = 18

    # Set some initial parameters for the whole plot
    plt.rcParams.update({'figure.figsize': (11,6), 'figure.dpi': 300, 'legend.framealpha': 0})
    figure, ax = plt.subplots(3, 1, sharex='all')
    plt.xlabel('Time (min)', fontsize=font_size_axes)
    plt.xlim((0, 5))
    figure.suptitle("Figure 4. Functional arm movements - IMU error", fontsize=font_size_title, weight='bold')

    # Create the torso subplot
    ax[0].scatter(time, T_diff_R1, s=1, c=thorax_colour)
    ax[0].scatter(time, T_diff_R2, s=1, c=thorax_colour)
    ax[0].scatter(time, T_diff_R3, s=1, c=thorax_colour)
    ax[0].scatter(time, T_diff_R4, s=1, c=thorax_colour)
    ax[0].scatter(time, T_diff_R5, s=1, c=thorax_colour)

    # Create the upper arm subplot
    ax[1].scatter(time, U_diff_R1, s=1, c=upper_colour)
    ax[1].scatter(time, U_diff_R2, s=1, c=upper_colour)
    ax[1].scatter(time, U_diff_R3, s=1, c=upper_colour)
    ax[1].scatter(time, U_diff_R4, s=1, c=upper_colour)
    ax[1].scatter(time, U_diff_R5, s=1, c=upper_colour)

    # Create the forearm subplot
    ax[2].scatter(time, F_diff_R1, s=1, c=forearm_colour)
    ax[2].scatter(time, F_diff_R2, s=1, c=forearm_colour)
    ax[2].scatter(time, F_diff_R3, s=1, c=forearm_colour)
    ax[2].scatter(time, F_diff_R4, s=1, c=forearm_colour)
    ax[2].scatter(time, F_diff_R5, s=1, c=forearm_colour)

    # Set some parameters for all the graphs
    for axs in ax.flat:
        axs.set(ylim=(0, y_range_max), yticks=range(0, 35, 10), axisbelow=True)
        axs.grid(axis="y", c=grid_colour)
        axs.tick_params(labelsize=font_size_ticks)

    # Set each graph title
    ax[0].annotate("Torso", (0.1, 32.5), fontsize=font_size_legend)
    ax[1].annotate("Upper Arm", (0.1, 32.5), fontsize=font_size_legend)
    ax[2].annotate("Forearm", (0.1, 32.5), fontsize=font_size_legend)

    # Set each graph legend
    ax[0].legend(["RMSE: " + str(round(torso_RMSE,1)) + " (" + str(round(torso_SD,1)) + ")" + chr(176)], loc="upper right", handletextpad=-2.0, handlelength=0, fontsize=font_size_legend)
    ax[1].legend(["RMSE: " + str(round(upper_RMSE,1)) + " (" + str(round(upper_SD,1)) + ")" + chr(176)], loc="upper right", handletextpad=-2.0, handlelength=0, fontsize=font_size_legend)
    ax[2].legend(["RMSE: " + str(round(forearm_RMSE,1)) + " (" + str(round(forearm_SD,1)) + ")" + chr(176)], loc="upper right", handletextpad=-2.0, handlelength=0, fontsize=font_size_legend)

    # Adjust the vertical space between plots
    plt.subplots_adjust(hspace=0.2)

    # Add common y axis label
    figure.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False, labelsize=font_size_ticks)
    plt.ylabel("IMU Error (" + chr(176) + ")", labelpad=10.0, fontsize=font_size_axes)

    plt.savefig("All_errors_vs_time.png")
    plt.clf()

