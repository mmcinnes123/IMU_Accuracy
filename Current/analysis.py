### Functions used to analyse the data

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from quat_functions import *


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
    plt.scatter(time, y2, s=3, c='fuchsia')
    plt.scatter(time, y3, s=3, c='red')
    plt.scatter(time, y4, s=3, c='maroon')
    plt.scatter(time, y5, s=3, c='blue')
    plt.scatter(time, y6, s=3, c='green')
    plt.scatter(time, y7, s=3, c='teal')
    plt.scatter(time, y8, s=3, c='darkgreen')
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
        q_diff = quaternion_multiply(quat_IMU, quaternion_conjugate(quat_Clust))
        q_diff_R = R.from_quat([q_diff[1], q_diff[2], q_diff[3], q_diff[0]])
        eul_diff = q_diff_R.as_euler(decomp_seq, degrees=True)
        diff_angle_1[row] = eul_diff[0]
        diff_angle_2[row] = eul_diff[1]
        diff_angle_3[row] = eul_diff[2]

    ### Calculate averages
    abs_diff_angle_1 = np.absolute(diff_angle_1)
    abs_diff_angle_2 = np.absolute(diff_angle_2)
    abs_diff_angle_3 = np.absolute(diff_angle_3)
    RMSD_angle_1 = (sum(np.square(diff_angle_1))/len(diff_angle_1))**0.5
    RMSD_angle_2 = (sum(np.square(diff_angle_2))/len(diff_angle_2))**0.5
    RMSD_angle_3 = (sum(np.square(diff_angle_3))/len(diff_angle_3))**0.5
    mean_angle_1 = np.mean(abs_diff_angle_1)
    mean_angle_2 = np.mean(abs_diff_angle_2)
    mean_angle_3 = np.mean(abs_diff_angle_3)
    sd_angle_1 = np.std(abs_diff_angle_1)
    sd_angle_2 = np.std(abs_diff_angle_2)
    sd_angle_3 = np.std(abs_diff_angle_3)

    return diff_angle_1, diff_angle_2, diff_angle_3, RMSD_angle_1, RMSD_angle_2, RMSD_angle_3


def find_smallest_angle(IMU_df, Clust_df):
    angle = np.zeros((len(IMU_df)))
    for row in range(len(IMU_df)):
        quat_mult = quaternion_multiply(IMU_df.values[row], quaternion_conjugate(Clust_df.values[row]))
        norm_vec_part = np.linalg.norm([quat_mult[1], quat_mult[2], quat_mult[3]])
        angle[row] = (2 * np.arcsin(norm_vec_part)) * 180 / np.pi
    RMSD_angle = (sum(np.square(angle))/len(angle))**0.5
    return angle, RMSD_angle


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
def plot_IMU_vs_Clus_eulers(IMU_eul_1, IMU_eul_2, IMU_eul_3, Clust_eul_1, Clust_eul_2, Clust_eul_3, angle, decomp_seq, tag, sample_rate):
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
    y7 = angle
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
    plt.savefig("IMUvsOMC_Eulers_Plot" + tag + "_" + decomp_seq + ".png")
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




