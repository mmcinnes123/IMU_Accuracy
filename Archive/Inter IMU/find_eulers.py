print("Running " + "find_eulers.py as " + __name__)

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import seaborn as sns
import logging


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

# Plot all the EUler angles of all IMUs and the NewAx cluster
def plot_all_eulers(IMU1_eul_1, IMU1_eul_2, IMU1_eul_3, IMU2_eul_1, IMU2_eul_2, IMU2_eul_3, IMU3_eul_1, IMU3_eul_2, IMU3_eul_3, NewAx_Clus_eul_1, NewAx_Clus_eul_2, NewAx_Clus_eul_3, sample_rate, decomp_seq, tag):
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
    y10 = NewAx_Clus_eul_1
    y11 = NewAx_Clus_eul_2
    y12 = NewAx_Clus_eul_3
    plt.scatter(x, y1, s=1, c='red')
    plt.scatter(x, y2, s=1, c='blue')
    plt.scatter(x, y3, s=1, c='green')
    plt.scatter(x, y4, s=1, c='red')
    plt.scatter(x, y5, s=1, c='blue')
    plt.scatter(x, y6, s=1, c='green')
    plt.scatter(x, y7, s=1, c='red')
    plt.scatter(x, y8, s=1, c='blue')
    plt.scatter(x, y9, s=1, c='green')
    plt.scatter(x, y10, s=1, c='darkred')
    plt.scatter(x, y11, s=1, c='darkblue')
    plt.scatter(x, y12, s=1, c='darkgreen')
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    plt.title("Euler Angles - Decomp Sequence: " + decomp_seq)
    plt.xlabel('Time')
    plt.ylabel('Degrees')
    # plt.grid(axis="x", which="both")
    # x_range = round(len(time)/sample_rate)
    # plt.xticks(range(0, x_range, 1))
    plt.legend(["IMU1 Angle 1", "IMU1 Angle 2", "IMU1 Angle 3", "IMU2 Angle 1", "IMU2 Angle 2", "IMU2 Angle 3", "IMU3 Angle 1", "IMU3 Angle 2", "IMU3 Angle 3", "Cluster Angle 1", "Cluster Angle 2", "Cluster Angle 3"], loc="lower left", markerscale=3)
    # plt.show()
    plt.savefig("Euler_Angle_Plot_" + tag + "_" + decomp_seq + ".png")
    plt.clf()




