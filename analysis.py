### Functions used to analyse the data

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from quat_functions import *
import scipy.signal


def adjust_angles_by_180(angle_arr):
    for i in range(len(angle_arr)):
        if angle_arr[i] < 0:
            angle_arr[i] = angle_arr[i] + 180
        else:
            angle_arr[i] = angle_arr[i] - 180
    return angle_arr

# Trim the data frames based on start and end time
def trim_df(df, start_time, end_time, sample_rate):
    first_index = int(start_time*sample_rate)
    last_index = int(end_time*sample_rate)
    index_range = list(range(first_index, last_index))
    df_new = df.iloc[index_range, :]
    df_new_new = df_new.reset_index(drop=True)
    return df_new_new

def read_abs_pre_processed_data_from_file(input_file):
    with open("ProcessedData/" + input_file, 'r') as file:
        df = pd.read_csv(file, header=0, sep=",")
    # Make seperate data frames
    IMU_df = df.filter(["IMU_Q0", "IMU_Q1", "IMU_Q2", "IMU_Q3"], axis=1)
    OMC_df = df.filter(["OMC_Q0", "OMC_Q1", "OMC_Q2", "OMC_Q3"], axis=1)
    return IMU_df, OMC_df

def read_inter_pre_processed_data_from_file(input_file):
    with open("ProcessedData/" + input_file, 'r') as file:
        df = pd.read_csv(file, header=0, sep=",")
    # Make seperate data frames
    IMU1_df = df.filter(["IMU1_Q0", "IMU1_Q1", "IMU1_Q2", "IMU1_Q3"], axis=1)
    IMU2_df = df.filter(["IMU2_Q0", "IMU2_Q1", "IMU2_Q2", "IMU2_Q3"], axis=1)
    IMU3_df = df.filter(["IMU3_Q0", "IMU3_Q1", "IMU3_Q2", "IMU3_Q3"], axis=1)
    IMU4_df = df.filter(["IMU4_Q0", "IMU4_Q1", "IMU4_Q2", "IMU4_Q3"], axis=1)
    return IMU1_df, IMU2_df, IMU3_df, IMU4_df

def cut_four_sections(df, s1, e1, s2, e2, s3, e3, s4, e4, sample_rate):
    index_range_1 = list(range(int(s1*sample_rate), int(e1*sample_rate)))
    index_range_2 = list(range(int(s2*sample_rate), int(e2*sample_rate)))
    index_range_3 = list(range(int(s3*sample_rate), int(e3*sample_rate)))
    index_range_4 = list(range(int(s4*sample_rate), int(e4*sample_rate)))
    df_1 = df.iloc[index_range_1, :]
    df_2 = df.iloc[index_range_2, :]
    df_3 = df.iloc[index_range_3, :]
    df_4 = df.iloc[index_range_4, :]
    final_df = pd.concat([df_1, df_2, df_3, df_4], ignore_index=True, sort=False)
    return final_df



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


def find_RMSD_four_IMUS(IMU1_single_angle_diff, IMU2_single_angle_diff, IMU3_single_angle_diff, IMU4_single_angle_diff):
    N = len(IMU1_single_angle_diff)
    RMSD = np.empty((N))
    for i in range(N):
        RMSD[i] = (np.mean(np.square([IMU1_single_angle_diff[i], IMU2_single_angle_diff[i], IMU3_single_angle_diff[i], IMU4_single_angle_diff[i]]))) ** 0.5
    return RMSD


# Plot the quaternions for comparison and checking timings
def plot_the_quats(IMU1_df, NewAx_Clus_df, tag, sample_rate):
    # Turn data frames into numpy arrays
    NewAx_Clus_array = NewAx_Clus_df.to_numpy(na_value="nan")
    IMU1_array = IMU1_df.to_numpy(na_value="nan")
    time = list(np.arange(0, len(NewAx_Clus_df) / sample_rate, 1 / sample_rate))
    if len(time) != len(NewAx_Clus_array):
        del time[-1]
    plt.figure(1)
    y1 = NewAx_Clus_array[:,0]
    y2 = NewAx_Clus_array[:,1]
    y3 = NewAx_Clus_array[:,2]
    y4 = NewAx_Clus_array[:,3]
    y5 = IMU1_array[:,0]
    y6 = IMU1_array[:,1]
    y7 = IMU1_array[:,2]
    y8 = IMU1_array[:,3]
    plt.scatter(time, y1, s=3, c='orange')
    plt.scatter(time, y2, s=3, c='firebrick')
    plt.scatter(time, y3, s=3, c='darkmagenta')
    plt.scatter(time, y4, s=3, c='green')
    plt.scatter(time, y5, s=3, c='gold')
    plt.scatter(time, y6, s=3, c='red')
    plt.scatter(time, y7, s=3, c='orchid')
    plt.scatter(time, y8, s=3, c='limegreen')
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    plt.title("Cluster and IMU2 Quaternions")
    plt.xlabel('Time')
    plt.ylabel('Quats')
    plt.grid(axis="x", which="both")
    x_range = round(len(time)/sample_rate)
    plt.xticks(range(0, x_range, 1))
    plt.legend(["Clus_Q0", "Clust_Q1", "Clust_Q2", "Clust_Q3", "IMU_Q0", "IMU_Q1", "IMU_Q2", "IMU_Q3"], loc="lower right")
    plt.savefig("RefQuats_" + tag + ".png")
    plt.clf()


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


def find_ang_dist(IMU_df, Clust_df):
    angle = np.zeros((len(IMU_df)))
    for row in range(len(IMU_df)):
        quat_mult = quat_mul(IMU_df.values[row], quat_conj(Clust_df.values[row]))
        norm_vec_part = np.linalg.norm([quat_mult[1], quat_mult[2], quat_mult[3]])
        angle[row] = (2 * np.arcsin(norm_vec_part)) * 180 / np.pi
    RMSD_angle = (sum(np.square(angle))/len(angle))**0.5
    return angle, RMSD_angle

def find_SD_quats(quat_arr, average_quat):
    ang_dist = np.zeros((len(quat_arr)))
    for row in range(len(quat_arr)):
        quat_mult = quat_mul(quat_arr[row], quat_conj(average_quat))
        norm_vec_part = np.linalg.norm([quat_mult[1], quat_mult[2], quat_mult[3]])
        ang_dist[row] = (2 * np.arcsin(norm_vec_part)) * 180 / np.pi
    variance = sum(ang_dist**2)/len(ang_dist)
    SD = variance**0.5
    return SD


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


# Plot all the Euler angles of all IMUs and the NewAx cluster
def plot_four_IMU_eulers(IMU1_eul_1, IMU1_eul_2, IMU1_eul_3, IMU2_eul_1, IMU2_eul_2, IMU2_eul_3,
                         IMU3_eul_1, IMU3_eul_2, IMU3_eul_3, IMU4_eul_1, IMU4_eul_2, IMU4_eul_3,
                         NewAx_Clus_eul_1, NewAx_Clus_eul_2, NewAx_Clus_eul_3, sample_rate, decomp_seq, tag):
    plt.figure(1)
    fig = plt.figure(figsize=(10, 8))
    time = list(np.arange(0, len(IMU1_eul_1) / sample_rate, 1 / sample_rate))
    x = time
    y1 = IMU1_eul_1
    y2 = IMU1_eul_2
    y3 = IMU1_eul_3
    y4 = IMU2_eul_1
    y5 = IMU2_eul_2
    y6 = IMU2_eul_3
    y7 = IMU3_eul_1
    y8 = IMU3_eul_2
    y9 = IMU3_eul_3
    y10 = IMU4_eul_1
    y11 = IMU4_eul_2
    y12 = IMU4_eul_3
    y13 = NewAx_Clus_eul_1
    y14 = NewAx_Clus_eul_2
    y15= NewAx_Clus_eul_3
    plt.scatter(x, y1, s=1, c='red')
    plt.scatter(x, y2, s=1, c='blue')
    plt.scatter(x, y3, s=1, c='green')
    plt.scatter(x, y4, s=1, c='red')
    plt.scatter(x, y5, s=1, c='blue')
    plt.scatter(x, y6, s=1, c='green')
    plt.scatter(x, y7, s=1, c='red')
    plt.scatter(x, y8, s=1, c='blue')
    plt.scatter(x, y9, s=1, c='green')
    plt.scatter(x, y10, s=1, c='red')
    plt.scatter(x, y11, s=1, c='blue')
    plt.scatter(x, y12, s=1, c='green')
    plt.scatter(x, y13, s=1, c='darkred')
    plt.scatter(x, y14, s=1, c='darkblue')
    plt.scatter(x, y15, s=1, c='darkgreen')
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    plt.title("4 IMU Euler Angles - Decomp Sequence: " + decomp_seq)
    plt.xlabel('Time')
    plt.ylabel('Degrees')
    # plt.grid(axis="x", which="both")
    # x_range = round(len(time)/sample_rate)
    # plt.xticks(range(0, x_range, 1))
    plt.legend(["IMU1 Angle 1", "IMU1 Angle 2", "IMU1 Angle 3", "IMU2 Angle 1", "IMU2 Angle 2", "IMU2 Angle 3",
                "IMU3 Angle 1", "IMU3 Angle 2", "IMU3 Angle 3", "IMU4 Angle 1", "IMU4 Angle 2", "IMU4 Angle 3",
                "Cluster Angle 1", "Cluster Angle 2", "Cluster Angle 3"], loc="lower left", markerscale=3)
    # plt.show()
    plt.savefig("All_Eulers_Plot" + tag + "_" + decomp_seq + ".png")
    plt.clf()

# Plot the Euler angles of one IMU and Cluster
def plot_IMU_vs_Clus_eulers(IMU_eul_1, IMU_eul_2, IMU_eul_3, Clust_eul_1, Clust_eul_2, Clust_eul_3, angular_dist, decomp_seq, tag, sample_rate):
    plt.figure(2)
    fig = plt.figure(figsize=(10, 8))
    time = list(np.arange(0, len(IMU_eul_1) / sample_rate, 1 / sample_rate))
    x = time
    y1 = IMU_eul_1
    y2 = IMU_eul_2
    y3 = IMU_eul_3
    y4 = Clust_eul_1
    y5 = Clust_eul_2
    y6 = Clust_eul_3
    y7 = angular_dist
    plt.scatter(x, y1, s=1, c='red')
    plt.scatter(x, y2, s=1, c='blue')
    plt.scatter(x, y3, s=1, c='green')
    plt.scatter(x, y4, s=1, c='darkred')
    plt.scatter(x, y5, s=1, c='darkblue')
    plt.scatter(x, y6, s=1, c='darkgreen')
    plt.scatter(x, y7, s=1, c='orange')
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    plt.title("IMU vs OMC Euler Angles - Decomp Sequence: " + decomp_seq)
    plt.xlabel('Time')
    plt.ylabel('Degrees')
    # plt.grid(axis="x", which="both")
    # x_range = round(len(time)/sample_rate)
    # plt.xticks(range(0, x_range, 1))
    plt.legend(["IMU1 Angle 1", "IMU1 Angle 2", "IMU1 Angle 3",
                "Cluster Angle 1", "Cluster Angle 2", "Cluster Angle 3", "Single-angle Difference"], loc="lower left", markerscale=3)
    # plt.show()
    plt.savefig("IMUvsOMC_Eulers_" + tag + "_" + decomp_seq + ".png")
    plt.clf()


# Find the angle of the projected local y vector on the chosen global plane (Cluster LCF):
def proj_vec_angle_global_Y(quat_df, global_axis):
    if global_axis == "Y":
        proj_angle = []
        # Horizontal 2D reference vector (unit vector along Z-axis, zero X-component:
        ref = np.array((0, -1))
        for row in range(len(quat_df)):
            # Vector to track around horizontal plane (the global X and Z components of the local y axis)
            rot_i = R.from_quat(list([quat_df.values[row, 1], quat_df.values[row, 2], quat_df.values[row, 3], quat_df.values[row, 0]]))
            rot_i_matrix = rot_i.as_matrix()
            proj_vec_i = np.array((rot_i_matrix[0,1], rot_i_matrix[2,1]))
            # Calculate angle between two vectors: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
            proj_angle_i = -np.arctan2((ref[0]*proj_vec_i[1] - ref[1]*proj_vec_i[0]), np.dot(ref, proj_vec_i))
            proj_angle.append(proj_angle_i * 180 / (np.pi))
        return proj_angle
    if global_axis == "X":
        proj_angle = []
        # Horizontal 2D reference vector (unit vector along Y-axis, zero Z-component:
        ref = np.array((1, 0))
        for row in range(len(quat_df)):
            # Vector to track around horizontal plane (the global X and Z components of the local y axis)
            rot_i = R.from_quat(list([quat_df.values[row, 1], quat_df.values[row, 2], quat_df.values[row, 3], quat_df.values[row, 0]]))
            rot_i_matrix = rot_i.as_matrix()
            proj_vec_i = np.array((rot_i_matrix[1,1], rot_i_matrix[2,1]))
            # Calculate angle between two vectors: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
            proj_angle_i = -np.arctan2((ref[0]*proj_vec_i[1] - ref[1]*proj_vec_i[0]), np.dot(ref, proj_vec_i))
            proj_angle.append(proj_angle_i * 180 / (np.pi))
        return proj_angle
    if global_axis == "Z":
        proj_angle = []
        # Horizontal 2D reference vector (unit vector along X-axis, zero Y-component:
        ref = np.array((1, 0))
        for row in range(len(quat_df)):
            # Vector to track around horizontal plane (the global X and Z components of the local y axis)
            rot_i = R.from_quat(list([quat_df.values[row, 1], quat_df.values[row, 2], quat_df.values[row, 3], quat_df.values[row, 0]]))
            rot_i_matrix = rot_i.as_matrix()
            proj_vec_i = np.array((rot_i_matrix[0,1], rot_i_matrix[1,1]))
            # Calculate angle between two vectors: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
            proj_angle_i = -np.arctan2((ref[0]*proj_vec_i[1] - ref[1]*proj_vec_i[0]), np.dot(ref, proj_vec_i))
            proj_angle.append(proj_angle_i * 180 / (np.pi))
        return proj_angle


# Plot the projected angle vector against time
def plot_proj_vec_angle(vector_angle, sample_rate, global_axis, tag):
    plt.figure(3)
    fig = plt.figure(figsize=(10, 8))
    time = list(np.arange(0, len(vector_angle) / sample_rate, 1 / sample_rate))
    x = time
    y1 = vector_angle
    plt.scatter(x, y1, s=1, c='red')
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    plt.title("Projector vector angle of OMC cluster - Around global axis: " + global_axis)
    plt.xlabel('Time')
    plt.ylabel('Degrees')
    plt.savefig("Vector_Angle_" + tag + ".png")
    plt.clf()


# Do a BA plot with the projected angle vector
def BA_plot(proj_vec_angle, diff_angle_1, tag):
    plt.figure(4)
    x = proj_vec_angle
    y1 = diff_angle_1
    plt.scatter(x, y1, s=1, c='red')
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    plt.title("BA Plot")
    plt.xlabel('OMC Projected Vector Angle [degrees]')
    plt.ylabel('Single-Angle Error [degrees]')
    plt.savefig("BAPlot" + tag + "_" + ".png")
    plt.clf()


# Do a BA plot with the euler angles. X-axis is cluster euler angle, Y-axis is difference. Can be Angle 1, 2 or 3
def BA_plot_eulers(IMU_Eul, Clust_Eul, which_angle, tag):
    eul_diff = []
    for row in range(len(IMU_Eul)):
        diff = abs(IMU_Eul[row] - Clust_Eul[row])
        eul_diff.append(diff)
    plt.figure(5)
    x = Clust_Eul
    y1 = eul_diff
    plt.scatter(x, y1, s=1, c='red')
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    plt.title("BA Plot - " + which_angle + " angle")
    plt.xlabel('OMC Euler Angle [degrees]')
    plt.ylabel('IMU-OMC Error in Euler Angle [degrees]')
    plt.savefig("BAPlot_Eul" + tag + "_" + ".png")
    plt.clf()


# Calculate the speed of the vector angle projection
def vec_angle_vel(proj_vec_angle, sample_rate):
    N = len(proj_vec_angle)
    ang_vel = []
    for i in range(N-1):
        ang_vel.append(sample_rate * (proj_vec_angle[i+1] - proj_vec_angle[i]))
    ang_vel.append(ang_vel[-1])
    return ang_vel


def BA_plot_combine_reps(angle_R1, diff_R1, angle_R2, diff_R2, angle_R3, diff_R3, label):
    all_angles = np.concatenate((angle_R1, angle_R2, angle_R3), axis=0)
    all_diffs = np.concatenate((diff_R1, diff_R2, diff_R3), axis=0)
    mean_diff = np.mean(all_diffs)
    sd_diff = np.std(all_diffs)
    CI_low = mean_diff - 1.96*sd_diff
    CI_high = mean_diff + 1.96*sd_diff
    plt.figure(5)
    x = all_angles
    y1 = all_diffs
    plt.scatter(x, y1, s=1, c='red')
    plt.axhline(mean_diff, color='black', linestyle='-')
    plt.axhline(CI_high, color='gray', linestyle='--')
    plt.axhline(CI_low, color='gray', linestyle='--')
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    xOutPlot = np.min(all_angles) + (np.max(all_angles) - np.min(all_angles)) * 1.14
    plt.text(xOutPlot, CI_low,
             r'-1.96SD:' + "\n" + "%.2f" % CI_low,
             ha="center",
             va="center",
             )
    plt.text(xOutPlot, CI_high,
             r'+1.96SD:' + "\n" + "%.2f" % CI_high,
             ha="center",
             va="center",
             )
    plt.text(xOutPlot, mean_diff,
             r'Mean:' + "\n" + "%.2f" % mean_diff,
             ha="center",
             va="center",
             )
    plt.subplots_adjust(right=0.85)

    plt.title("BA Plot - All Reps Combined")
    plt.xlabel('OMC Projected Vector Angle [degrees]')
    plt.ylabel('Single-Angle Error [degrees]')
    plt.savefig("BAPlot_combined" + label + "_" + ".png")
    plt.clf()



def plot_error_vs_time(single_angle_diff, tag):
    time = np.array(list(np.arange(0, len(single_angle_diff) / 100, 1 / 100)))
    plt.figure(6)
    x = time
    y1 = single_angle_diff
    plt.scatter(x, y1, s=1, c='red')

    # obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(x, y1, 1)
    # add linear regression line to scatterplot
    y2 = m * x + b
    plt.plot(x,y2 )

    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    plt.title("Error vs Time: " + tag)
    plt.xlabel('Time')
    plt.ylabel('Angular Distance [degrees]')
    # plt.grid(axis="x", which="both")
    # x_range = round(len(time)/sample_rate)
    plt.ylim((0,30))
    # plt.xticks(range(0, x_range, 1))
    plt.legend(["Single-angle Difference"], loc="lower left", markerscale=3)
    # plt.show()
    plt.savefig("Error_vs_Time_" + tag + ".png")
    plt.clf()


    return corr_coeff


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


def find_joint_rot_quat(q_df_prox, q_df_dist):
    N = len(q_df_dist)
    q_joint_arr = np.zeros((N, 4))
    for row in range(N):
        q_prox_i = np.array([q_df_prox.values[row, 0], q_df_prox.values[row,1], q_df_prox.values[row,2], q_df_prox.values[row,3]])
        q_dist_i = np.array([q_df_dist.values[row, 0], q_df_dist.values[row,1], q_df_dist.values[row,2], q_df_dist.values[row,3]])
        q_joint_arr[row] = quat_mul(quat_conj(q_prox_i), q_dist_i)
    q_joint_df = pd.DataFrame(q_joint_arr)
    return q_joint_df


def find_euler_error(IMU_angles, OMC_angles):
    error = abs(IMU_angles - OMC_angles)
    # If error is over 100 degrees, remove from the array
    error = error[error < 300]
    RMSD = (sum(np.square(error)) / len(IMU_angles)) ** 0.5
    return RMSD

# Flip negatives values so that transfer from 180 to -180 becomes continuous on positive scale
def change_neg_angles(angle_arr):
    for i in range(len(angle_arr)):
        if angle_arr[i] < 0:
            angle_arr[i] = angle_arr[i] + 360
        else:
            angle_arr[i] = angle_arr[i]
    return


def find_smallest_angle_new(IMU_df, Clust_df):
    angle = np.zeros((len(IMU_df)))
    for row in range(len(IMU_df)):
        quat1 = IMU_df.values[row]
        quat2 = Clust_df.values[row]
        angle[row] = 2 * np.arccos(abs(np.inner(quat1, quat2))) * 180 / np.pi
    RMSD_angle = (sum(np.square(angle))/len(angle))**0.5
    return angle, RMSD_angle


def find_corr_coef(single_angle_diff):
    time = np.array(list(np.arange(0, len(single_angle_diff) / 100, 1 / 100)))
    corr_coeff = np.corrcoef(single_angle_diff, time)[0,1]
    return corr_coeff


def plot_all_error_vs_time(single_angle_diff_R1, single_angle_diff_R2, single_angle_diff_R3, single_angle_diff_R4, single_angle_diff_R5, average_corr_coeff, average_SD_corr_coeff, which_limb):
    time = np.array(list(np.arange(0, len(single_angle_diff_R1) / 100, 1 / 100)))
    plt.figure(6)
    x = time
    y1 = single_angle_diff_R1
    y2 = single_angle_diff_R2
    y3 = single_angle_diff_R3
    y4 = single_angle_diff_R4
    y5 = single_angle_diff_R5
    plt.scatter(x, y1, s=0.5, c='lightcoral')
    plt.scatter(x, y2, s=0.5, c='lightcoral')
    plt.scatter(x, y3, s=0.5, c='lightcoral')
    plt.scatter(x, y4, s=0.5, c='lightcoral')
    plt.scatter(x, y5, s=0.5, c='lightcoral')


    # obtain m (slope) and b(intercept) of linear regression line
    m_1, b_1 = np.polyfit(x, y1, 1)
    m_2, b_2 = np.polyfit(x, y2, 1)
    m_3, b_3 = np.polyfit(x, y3, 1)
    m_4, b_4 = np.polyfit(x, y4, 1)
    m_5, b_5 = np.polyfit(x, y5, 1)

    av_m = np.mean([m_1, m_2, m_3, m_4, m_5])
    av_b = np.mean([b_1, b_2, b_3, b_4, b_5])
    # Average regression line
    lin = av_m*x + av_b
    plt.plot(x, lin, c='maroon', linewidth=2)

    text_x = 0.7*np.max(time)
    text_y = 20
    plt.text(text_x, text_y,
             r'Pearsons Correlation Coefficient: ' + '\n'
             + '(mean (SD))' + '\n' +
             ''"%.2f" % average_corr_coeff + ' (' + "%.2f" % average_SD_corr_coeff + ')',
             ha="center",
             va="center",
             )

    plt.rcParams.update({'figure.figsize': (20, 16), 'figure.dpi': 300})
    plt.title(which_limb + " IMU Error")
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Distance [degrees]')
    # plt.grid(axis="x", which="both")
    # x_range = round(len(time)/sample_rate)
    plt.ylim((0, 30))
    # plt.xticks(range(0, x_range, 1))
    # plt.legend(["R"], loc="lower left", markerscale=3)
    # plt.show()
    plt.savefig("Error_vs_Time_" + which_limb + ".png")
    plt.clf()


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