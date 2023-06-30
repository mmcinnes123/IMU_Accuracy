### Functions to find a rotation quaternion from IMU GCF to OMC GCF
    # Angular velocity vector are extracted from quaternion data
    # A root() function is used to find a rotation quaternion based an a dataframe of adjacent IMU and cluster velocity vector
    # The resultant quaternion is then applied to the IMU data


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import pandas as pd
from quat_functions import *

# Calculate the three-element angular velocity vectors in the LCF from a dataframe of quaternions
def find_ang_vels_local(df):
    N = len(df)-1
    ang_vels = np.empty((0,3))
    for i in range(N):
        quat1 = df.values[i]
        quat2 = df.values[i+1]
        if np.isnan(quat1).any(axis=0) == False and np.isnan(quat1).any(axis=0) == False:
            ang_vel_i = ang_vel_from_quats(quat1, quat2, 0.01)
            ang_vels = np.append(ang_vels, np.array([ang_vel_i]), axis=0)

    M = len(ang_vels)
    ang_vels_mag = np.zeros((M))
    for i in range(M):
        vel_mag_i = (ang_vels[i, 0]**2 + ang_vels[i, 1]**2 + ang_vels[i, 2]**2)**0.5
        ang_vels_mag[i] = vel_mag_i

    return ang_vels, ang_vels_mag


# Calculate the three-element angular velocity vectors in the GCF from a dataframe of quaternions
def find_ang_vels_global(df):
    N = len(df) - 11
    ang_vels = np.empty((0, 3))
    for i in range(N):
        quat1 = df.values[i]
        quat2 = df.values[i + 10]
        if np.isnan(quat1).any(axis=0) == False and np.isnan(quat1).any(axis=0) == False:
            ang_vel_local_i = ang_vel_from_quats(quat1, quat2, 0.1)
            ang_vel_local_i_asquat = np.concatenate(([0], ang_vel_local_i))
            ang_vel_global_i_asquat = quaternion_multiply(quaternion_multiply(quat1, ang_vel_local_i_asquat), quaternion_conjugate(quat1))
            ang_vel_global_i = np.delete(ang_vel_global_i_asquat, 0)
            ang_vels = np.append(ang_vels, np.array([ang_vel_global_i]), axis=0)
    M = len(ang_vels)
    ang_vels_mag = np.zeros((M))
    for i in range(M):
        vel_mag_i = (ang_vels[i, 0] ** 2 + ang_vels[i, 1] ** 2 + ang_vels[i, 2] ** 2) ** 0.5
        ang_vels_mag[i] = vel_mag_i

    return ang_vels, ang_vels_mag


# Plot the magnitudes of the angular velocity vectors
def plot_ang_vels_mag(vel_array1, vel_array2, tag):
    sample_rate = 100
    time = list(np.arange(0, len(vel_array1) / sample_rate, 1 / sample_rate))
    if len(time) != len(vel_array1):
        del time[-1]
    plt.figure(1)
    y1 = vel_array1
    y2 = vel_array2
    plt.scatter(time, y1, s=3, c='orange')
    plt.scatter(time, y2, s=3, c='blue')
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    plt.title("Angular velocity")
    plt.xlabel('Time')
    plt.ylabel('Rad/s')
    plt.grid(axis="x", which="both")
    plt.ylim(-10,10)
    x_range = round(len(time)/sample_rate)
    plt.xticks(range(0, x_range, 1))
    plt.legend(["IMU angular velocity", "Cluster angular velocity"], loc="lower right")
    plt.savefig("Ang_vel_" + tag + ".png")
    plt.clf()


# Plot the xyz componenets of the angular velocity vectors
def plot_ang_vels_xyz(IMU_ang_vels, Clus_ang_vels, tag):
    wx_I = IMU_ang_vels[:, 0]
    wy_I = IMU_ang_vels[:, 1]
    wz_I = IMU_ang_vels[:, 2]
    wx_M = Clus_ang_vels[:, 0]
    wy_M = Clus_ang_vels[:, 1]
    wz_M = Clus_ang_vels[:, 2]
    sample_rate = 100
    time = list(np.arange(0, len(IMU_ang_vels) / sample_rate, 1 / sample_rate))
    if len(time) != len(IMU_ang_vels):
        del time[-1]
    y1 = wx_I
    y2 = wy_I
    y3 = wz_I
    y4 = wx_M
    y5 = wy_M
    y6 = wz_M
    plt.scatter(time, y1, s=3, c='orange')
    plt.scatter(time, y2, s=3, c='blue')
    plt.scatter(time, y3, s=3, c='green')
    plt.scatter(time, y4, s=3, c='red')
    plt.scatter(time, y5, s=3, c='darkblue')
    plt.scatter(time, y6, s=3, c='darkgreen')
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    plt.title("Angular velocity components")
    plt.xlabel('Time')
    plt.ylabel('Rad/s')
    plt.grid(axis="x", which="both")
    x_range = round(len(time)/sample_rate)
    plt.xticks(range(0, x_range, 1))
    plt.legend(["IMU w_x", "IMU w_y", "IMU w_z", "Cluster w_x", "Cluster w_y", "Cluster w_z"], loc="lower right")
    plt.savefig("Ang_vel_xyz" + tag + ".png")
    plt.clf()

    wx_diff = (wx_I - wx_M)
    wy_diff = (wy_I - wy_M)
    wz_diff = (wz_I - wz_M)
    y7 = wx_diff
    y8 = wy_diff
    y9 = wz_diff
    plt.scatter(time, y7, s=3, c='orange')
    plt.scatter(time, y8, s=3, c='blue')
    plt.scatter(time, y9, s=3, c='green')
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    plt.title("Difference in Angular velocity components")
    plt.xlabel('Time')
    plt.ylabel('Rad/s')
    plt.grid(axis="x", which="both")
    x_range = round(len(time)/sample_rate)
    plt.xticks(range(0, x_range, 1))
    plt.legend(["Diff w_x", "Diff w_y", "Diff w_z"], loc="lower right")
    plt.savefig("Ang_vel_xyz_diff" + tag + ".png")
    plt.clf()

    print(np.mean(wx_diff))
    print(np.mean(wy_diff))
    print(np.mean(wz_diff))


# Combine the angular velocities into one array, normalise, and trim based on max and min angular velocity
def limit_vel_range(IMU_ang_vels, Clus_ang_vels, min_vel, max_vel):
    ang_vels = np.concatenate((IMU_ang_vels, Clus_ang_vels), axis=1)
    ang_vels_trimmed = np.empty((0,6))
    for i in range(len(ang_vels)):
        if min_vel < np.linalg.norm(ang_vels[i,0:3]) < max_vel and min_vel < np.linalg.norm(ang_vels[i,3:6]) < max_vel:
            IMU_ang_vel_i = [ang_vels[i,0:3]/np.linalg.norm(ang_vels[i,0:3])]
            Clust_ang_vel_i = [ang_vels[i,3:6]/np.linalg.norm(ang_vels[i,3:6])]
            ang_vel_i = np.concatenate((IMU_ang_vel_i, Clust_ang_vel_i), axis=1)
            ang_vels_trimmed = np.append(ang_vels_trimmed, np.array(ang_vel_i), axis=0)
    return ang_vels_trimmed


# Find the rotation quaternion from IMU to OMC GCF based on angular velocity vectors (from combined vectors in a N x 6 df)
def vel_rot_quat(ang_vels):

    weight = 10.0   # This gives a weight to the constraint that the resultant quat is normalised
    wI_x = ang_vels[:,0]
    wI_y = ang_vels[:,1]
    wI_z = ang_vels[:,2]
    wM_x = ang_vels[:,3]
    wM_y = ang_vels[:,4]
    wM_z = ang_vels[:,5]

    # Function built from system of non-linear equations
    def f(wxyz):
        q0 = wxyz[0]
        q1 = wxyz[1]
        q2 = wxyz[2]
        q3 = wxyz[3]
        for i in range(len(ang_vels)):
            f1 = (wM_x[i] - wI_x[i]) * q0 - (wM_z[i] + wI_z[i]) * q2 + (wM_y[i] + wI_y[i]) * q3
            f2 = (wM_y[i] - wI_y[i]) * q0 + (wM_z[i] + wI_z[i]) * q1 - (wM_x[i] + wI_x[i]) * q3
            f3 = (wM_z[i] - wI_z[i]) * q0 - (wM_y[i] + wI_y[i]) * q1 + (wM_x[i] + wI_x[i]) * q2
            f4 = weight * (q0**2 + q1**2 + q2**2 + q3**2 - 1)
        return np.array([f1, f2, f3, f4])

    wxyz_0 = np.array([1.0, 0, 0, 0])   # Give the functions a starting point quat
    wxyz = root(f, wxyz_0, method='lm')     # Solve using Levenberg-Marquardt method

    if wxyz.success == True:
        wxyz_norm = wxyz.x / np.linalg.norm(wxyz.x)
    else:
        print("ERROR: root() failed")
        wxyz_norm = [1, 0, 0, 0]

    print(wxyz)
    print("Rotation quaternion:")
    print(round(wxyz_norm[0], 4), round(wxyz_norm[1], 4), round(wxyz_norm[2], 4), round(wxyz_norm[3], 4))

    rot_2_apply = wxyz_norm

    return rot_2_apply


# Apply the GCF velocity-vector rotation quaternion to global IMU data
def trans_clust_vel_GCF(clus_df, rot_2_apply):
    N = len(clus_df)
    new_clus = np.zeros((N,4))
    for row in range(N):
        quat_i = np.array([clus_df.values[row, 0], clus_df.values[row, 1], clus_df.values[row, 2], clus_df.values[row, 3]])
        new_clus[row] = quaternion_multiply(quaternion_multiply(rot_2_apply, quat_i), quaternion_conjugate(rot_2_apply))
    new_clus_df = pd.DataFrame(new_clus)
    return new_clus_df

