# This script can perform several types of analysis on a comparison between IMU and OMC Cluster data.
# 1. 'Vector Angle' calculates a 2D projected vector of an chosen local coordinate axis for both IMU and Cluster
# and measures the change in rotation of that axis relative to a specified global axis. OptiTrack cluster should be used.
# 2. 'Eulers' simply convert the quaternion orientations into euler angles then calculate the difference between IMU and Cluster.
# 3. 'Quat Difference' calculates a rotation quaternion between the IMU and Clust at every point in time and then decomposes that into Euler angles
# 4. 'Smallest Angle' calculates the smallest angle between the IMU and Cluster quaternions (1 variable output)
#
# Input: a .csv file with the transformed data
# Output: a 'Results' log, and different plots dependent on the chosen analysis method

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import seaborn as sns
import logging

### SETTINGS

input_file = "MP_H_R5_transformed.csv"
# Time interval of interest:
start_time = 3
end_time = 30
# Which cluster quats? OptiTrack or New_ax
Which_clust = "New_ax"
# Global Axis of interest:
glob_axis = "Y"
# Choose which angles you want to calculate and plot:
#(For horizontal plane movements, choose IMU_x_Y and Clust_x_Y)
find_angle_IMU_x_Y = True
find_angle_IMU_y_Y = False
find_angle_IMU_z_Y = False
find_angle_IMU_x_X = False
find_angle_IMU_y_X = False
find_angle_IMU_z_X = False
find_angle_IMU_x_Z = False
find_angle_IMU_y_Z = False
find_angle_IMU_z_Z = False
find_angle_Clust_x_Y = True
find_angle_Clust_y_Y = False
find_angle_Clust_z_Y = False
find_angle_Clust_x_X = False
find_angle_Clust_y_X = False
find_angle_Clust_z_X = False
find_angle_Clust_x_Z = False
find_angle_Clust_y_z = False
find_angle_Clust_z_Z = False
# Choose local axes of interest in the code

# Chose Euler Angle decomp sequence
decomp_seq = "XYZ"

# Which type of analysis?
plot_vector_angle = False
plot_eulers = True
plot_quat_difference = True
plot_smallest_angle = True

# Set up
sample_rate = 100
skip_rows = start_time*sample_rate
num_rows = (end_time-start_time)*sample_rate
tag = input_file.replace("_transformed.csv", "")
# output_file_name = tag + "_" + "_" + "diff.csv"
vec_angle_output_plot_file = "Vector_angles_" + tag + ".png"

logging.basicConfig(filename='Results.log', level=logging.INFO)
logging.info("INPUT FILE: " + input_file)
logging.info("Global axis of interest: " + glob_axis)
logging.info("Decomp sequence: " + decomp_seq)
logging.info("Time Interval: " + str(start_time) + " to " + str(end_time))
logging.info("Analysis Method: Vector Angle? " + str(plot_vector_angle))
logging.info("Analysis Method: Euler Angles? " + str(plot_eulers))
logging.info("Analysis Method: Quaternion Difference in Euler Angles? " + str(plot_quat_difference))
logging.info("Analysis Method: Smallest Angle Between Quats? " + str(plot_smallest_angle))

### DEFINE FUNCTIONS

def quaternion_multiply(Q0, Q1):
    """
    Multiplies two quaternions.
    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31)
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32)
    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)
    """
    w0 = Q0[0]
    x0 = Q0[1]
    y0 = Q0[2]
    z0 = Q0[3]
    w1 = Q1[0]
    x1 = Q1[1]
    y1 = Q1[2]
    z1 = Q1[3]
    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])
    return final_quaternion

def quaternion_conjugate(Q0):
    w0 = Q0[0]
    x0 = Q0[1]
    y0 = Q0[2]
    z0 = Q0[3]
    output_quaternion = np.array([w0, -x0, -y0, -z0])
    return output_quaternion

## READ IN DATA

# Read in the first IMU and cluster data to get reference position
with open(input_file, 'r') as file:
    columns_IMU = [0, 1, 2, 3]
    IMU_ref_df = pd.read_csv(file, usecols=columns_IMU, nrows=1)
    
with open(input_file, 'r') as file:
    if Which_clust == "OptiTrack":
        columns_Clust = [4, 5, 6, 7]
    elif Which_clust == "New_ax":
        columns_Clust = [8, 9, 10, 11]
    Clust_ref_df = pd.read_csv(file, usecols=columns_Clust, nrows=1)

# Read in IMU data
with open(input_file, 'r') as file:
    columns_IMU = [0, 1, 2, 3]
    IMU_df = pd.read_csv(file, usecols=columns_IMU, skiprows=skip_rows, nrows=num_rows)

# Read in Cluster data
with open(input_file, 'r') as file:
    if Which_clust == "OptiTrack":
        columns_Clust = [4, 5, 6, 7]
    elif Which_clust == "New_ax":
        columns_Clust = [8, 9, 10, 11]
    Clust_df = pd.read_csv(file, usecols=columns_Clust, skiprows=skip_rows, nrows=num_rows)

# Make a time column
time = list(np.arange(0, len(IMU_df)/sample_rate, 1/sample_rate))

### CALCULATE AND PLOT VECTOR ANGLES

if plot_vector_angle == True:

    ### CREATE REFERENCE VECTORS

    ## From IMU Data

    # Create a reference orientation matrices from the first time sample (from which all other vectors will be compared)
    IMU_ref_R = R.from_quat(
        list([IMU_ref_df.values[0, 1], IMU_ref_df.values[0, 2], IMU_ref_df.values[0, 3], IMU_ref_df.values[0, 0]]))
    IMU_ref = np.array(IMU_ref_R.as_matrix())

    # Create reference vectors for each local axis, and for each global axis of interest
    # For rotation around global Y (looking at X and Z elements)
    x_XZ_ref_IMU = np.array((IMU_ref[0, 0], IMU_ref[2, 0]))
    y_XZ_ref_IMU = np.array((IMU_ref[0, 1], IMU_ref[2, 1]))
    z_XZ_ref_IMU = np.array((IMU_ref[0, 2], IMU_ref[2, 2]))
    # For rotation around global X (looking at Y and Z elements)
    x_YZ_ref_IMU = np.array((IMU_ref[1, 0], IMU_ref[2, 0]))
    y_YZ_ref_IMU = np.array((IMU_ref[1, 1], IMU_ref[2, 1]))
    z_YZ_ref_IMU = np.array((IMU_ref[1, 2], IMU_ref[2, 2]))
    # For rotation around global Z (looking at X and Y elements)
    x_XY_ref_IMU = np.array((IMU_ref[0, 0], IMU_ref[1, 0]))
    y_XY_ref_IMU = np.array((IMU_ref[0, 1], IMU_ref[1, 1]))
    z_XY_ref_IMU = np.array((IMU_ref[0, 2], IMU_ref[1, 2]))

    ## From Clust Data

    # Create a reference orientation matrices from the first time sample (from which all other vectors will be compared)
    Clust_ref_R = R.from_quat(
        list([Clust_ref_df.values[0, 1], Clust_ref_df.values[0, 2], Clust_ref_df.values[0, 3], Clust_ref_df.values[0, 0]]))
    Clust_ref = np.array(Clust_ref_R.as_matrix())

    # Create reference vectors for each local axis, and for each global axis of interest
    # For rotation around global Y (looking at X and Z elements)
    x_XZ_ref_Clust = np.array((Clust_ref[0, 0], Clust_ref[2, 0]))
    y_XZ_ref_Clust = np.array((Clust_ref[0, 1], Clust_ref[2, 1]))
    z_XZ_ref_Clust = np.array((Clust_ref[0, 2], Clust_ref[2, 2]))
    # For rotation around global X (looking at Y and Z elements)
    x_YZ_ref_Clust = np.array((Clust_ref[1, 0], Clust_ref[2, 0]))
    y_YZ_ref_Clust = np.array((Clust_ref[1, 1], Clust_ref[2, 1]))
    z_YZ_ref_Clust = np.array((Clust_ref[1, 2], Clust_ref[2, 2]))
    # For rotation around global Z (looking at X and Y elements)
    x_XY_ref_Clust = np.array((Clust_ref[0, 0], Clust_ref[1, 0]))
    y_XY_ref_Clust = np.array((Clust_ref[0, 1], Clust_ref[1, 1]))
    z_XY_ref_Clust = np.array((Clust_ref[0, 2], Clust_ref[1, 2]))

    IMU_angle_x_Y = []
    IMU_angle_y_Y = []
    IMU_angle_z_Y = []
    IMU_angle_x_X = []
    IMU_angle_y_X = []
    IMU_angle_z_X = []
    IMU_angle_x_Z = []
    IMU_angle_y_Z = []
    IMU_angle_z_Z = []

    Clust_angle_x_Y = []
    Clust_angle_y_Y = []
    Clust_angle_z_Y = []
    Clust_angle_x_X = []
    Clust_angle_y_X = []
    Clust_angle_z_X = []
    Clust_angle_x_Z = []
    Clust_angle_y_Z = []
    Clust_angle_z_Z = []

    ### CALCULATE ANGLE CHANGE AROUND GLOBAL AXIS OF INTEREST

    # For every time sample in the data frame, extract the right vector from the DCM and calculate angle difference around the axis of interest

    if glob_axis == "Y":
        # From IMU data
        for row in range(len(IMU_df)):
            # Don't calculate it for first value since vector 1 = vector 2 and arccos function can't solve
            if row == 0:
                IMU_angle_x_Y.append(0)
                IMU_angle_y_Y.append(0)
                IMU_angle_z_Y.append(0)
            else:
                IMU_i_R = R.from_quat(
                    list([IMU_df.values[row, 1], IMU_df.values[row, 2], IMU_df.values[row, 3], IMU_df.values[row, 0]]))
                IMU_i = IMU_i_R.as_matrix()
                x_XZ_i = np.array((IMU_i[0, 0], IMU_i[2, 0]))
                y_XZ_i = np.array((IMU_i[0, 1], IMU_i[2, 1]))
                z_XZ_i = np.array((IMU_i[0, 2], IMU_i[2, 2]))
                # Apply quation: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
                if find_angle_IMU_x_Y == True:
                    angle_x_Y_i = np.arccos(
                        np.dot(x_XZ_ref_IMU, x_XZ_i) / (np.linalg.norm(x_XZ_ref_IMU) * np.linalg.norm(x_XZ_i)))
                    IMU_angle_x_Y.append(angle_x_Y_i * 180 / (np.pi))
                if find_angle_IMU_y_Y == True:
                    angle_y_Y_i = np.arccos(
                        np.dot(y_XZ_ref_IMU, y_XZ_i) / (np.linalg.norm(y_XZ_ref_IMU) * np.linalg.norm(y_XZ_i)))
                    IMU_angle_y_Y.append(angle_y_Y_i * 180 / (np.pi))
                if find_angle_IMU_z_Y == True:
                    angle_z_Y_i = np.arccos(
                        np.dot(z_XZ_ref_IMU, z_XZ_i) / (np.linalg.norm(z_XZ_ref_IMU) * np.linalg.norm(z_XZ_i)))
                    IMU_angle_z_Y.append(angle_z_Y_i * 180 / (np.pi))

        # From Clust data
        for row in range(len(Clust_df)):
            # Don't calculate it for first value since vector 1 = vector 2 and arccos function can't solve
            if row == 0:
                Clust_angle_x_Y.append(0)
                Clust_angle_y_Y.append(0)
                Clust_angle_z_Y.append(0)
            else:
                Clust_i_R = R.from_quat(list(
                    [Clust_df.values[row, 1], Clust_df.values[row, 2], Clust_df.values[row, 3], Clust_df.values[row, 0]]))
                Clust_i = Clust_i_R.as_matrix()
                x_XZ_i = np.array((Clust_i[0, 0], Clust_i[2, 0]))
                y_XZ_i = np.array((Clust_i[0, 1], Clust_i[2, 1]))
                z_XZ_i = np.array((Clust_i[0, 2], Clust_i[2, 2]))
                # Apply quation: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
                if find_angle_Clust_x_Y == True:
                    angle_x_Y_i = np.arccos(
                        np.dot(x_XZ_ref_Clust, x_XZ_i) / (np.linalg.norm(x_XZ_ref_Clust) * np.linalg.norm(x_XZ_i)))
                    Clust_angle_x_Y.append(angle_x_Y_i * 180 / (np.pi))
                if find_angle_Clust_y_Y == True:
                    angle_y_Y_i = np.arccos(
                        np.dot(y_XZ_ref_Clust, y_XZ_i) / (np.linalg.norm(y_XZ_ref_Clust) * np.linalg.norm(y_XZ_i)))
                    Clust_angle_y_Y.append(angle_y_Y_i * 180 / (np.pi))
                if find_angle_Clust_z_Y == True:
                    angle_z_Y_i = np.arccos(
                        np.dot(z_XZ_ref_Clust, z_XZ_i) / (np.linalg.norm(z_XZ_ref_Clust) * np.linalg.norm(z_XZ_i)))
                    Clust_angle_z_Y.append(angle_z_Y_i * 180 / (np.pi))

    if glob_axis == "X":

        # From IMU data
        for row in range(len(IMU_df)):
            # Don't calculate it for first value since vector 1 = vector 2 and arccos function can't solve
            if row == 0:
                IMU_angle_x_X.append(0)
                IMU_angle_y_X.append(0)
                IMU_angle_z_X.append(0)
            else:
                IMU_i_R = R.from_quat(
                    list([IMU_df.values[row, 1], IMU_df.values[row, 2], IMU_df.values[row, 3], IMU_df.values[row, 0]]))
                IMU_i = IMU_i_R.as_matrix()
                x_YZ_i = np.array((IMU_i[1, 0], IMU_i[2, 0]))
                y_YZ_i = np.array((IMU_i[1, 1], IMU_i[2, 1]))
                z_YZ_i = np.array((IMU_i[1, 2], IMU_i[2, 2]))
                # Apply equation: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
                if find_angle_IMU_x_X == True:
                    angle_x_X_i = np.arccos(
                        np.dot(x_YZ_ref_IMU, x_YZ_i) / (np.linalg.norm(x_YZ_ref_IMU) * np.linalg.norm(x_YZ_i)))
                    IMU_angle_x_X.append(angle_x_X_i * 180 / (np.pi))
                if find_angle_IMU_y_X == True:
                    angle_y_X_i = np.arccos(
                        np.dot(y_YZ_ref_IMU, y_YZ_i) / (np.linalg.norm(y_YZ_ref_IMU) * np.linalg.norm(y_YZ_i)))
                    IMU_angle_y_X.append(angle_y_X_i * 180 / (np.pi))
                if find_angle_IMU_z_X == True:
                    angle_z_X_i = np.arccos(
                        np.dot(z_YZ_ref_IMU, z_YZ_i) / (np.linalg.norm(z_YZ_ref_IMU) * np.linalg.norm(z_YZ_i)))
                    IMU_angle_z_X.append(angle_z_X_i * 180 / (np.pi))

        # From Clust data
        for row in range(len(Clust_df)):
            # Don't calculate it for first value since vector 1 = vector 2 and arccos function can't solve
            if row == 0:
                Clust_angle_x_X.append(0)
                Clust_angle_y_X.append(0)
                Clust_angle_z_X.append(0)
            else:
                Clust_i_R = R.from_quat(list(
                    [Clust_df.values[row, 1], Clust_df.values[row, 2], Clust_df.values[row, 3], Clust_df.values[row, 0]]))
                Clust_i = Clust_i_R.as_matrix()
                x_YZ_i = np.array((Clust_i[1, 0], Clust_i[2, 0]))
                y_YZ_i = np.array((Clust_i[1, 1], Clust_i[2, 1]))
                z_YZ_i = np.array((Clust_i[1, 2], Clust_i[2, 2]))
                # Apply equation: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
                if find_angle_Clust_x_X == True:
                    angle_x_X_i = np.arccos(
                        np.dot(x_YZ_ref_Clust, x_YZ_i) / (np.linalg.norm(x_YZ_ref_Clust) * np.linalg.norm(x_YZ_i)))
                    Clust_angle_x_X.append(angle_x_X_i * 180 / (np.pi))
                if find_angle_Clust_y_X == True:
                    angle_y_X_i = np.arccos(
                        np.dot(y_YZ_ref_Clust, y_YZ_i) / (np.linalg.norm(y_YZ_ref_Clust) * np.linalg.norm(y_YZ_i)))
                    Clust_angle_y_X.append(angle_y_X_i * 180 / (np.pi))
                if find_angle_Clust_z_X == True:
                    angle_z_X_i = np.arccos(
                        np.dot(z_YZ_ref_Clust, z_YZ_i) / (np.linalg.norm(z_YZ_ref_Clust) * np.linalg.norm(z_YZ_i)))
                    Clust_angle_z_X.append(angle_z_X_i * 180 / (np.pi))

    if glob_axis == "Z":

        # From IMU data
        for row in range(len(IMU_df)):
            # Don't calculate it for first value since vector 1 = vector 2 and arccos function can't solve
            if row == 0:
                IMU_angle_x_Z.append(0)
                IMU_angle_y_Z.append(0)
                IMU_angle_z_Z.append(0)
            else:
                IMU_i_R = R.from_quat(
                    list([IMU_df.values[row, 1], IMU_df.values[row, 2], IMU_df.values[row, 3], IMU_df.values[row, 0]]))
                IMU_i = IMU_i_R.as_matrix()
                x_XY_i = np.array((IMU_i[0, 0], IMU_i[1, 0]))
                y_XY_i = np.array((IMU_i[0, 1], IMU_i[1, 1]))
                z_XY_i = np.array((IMU_i[0, 2], IMU_i[1, 2]))
                # Apply equation: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
                if find_angle_IMU_x_Z == True:
                    angle_x_Z_i = np.arccos(
                        np.dot(x_XY_ref_IMU, x_XY_i) / (np.linalg.norm(x_XY_ref_IMU) * np.linalg.norm(x_XY_i)))
                    IMU_angle_x_Z.append(angle_x_Z_i * 180 / (np.pi))
                if find_angle_IMU_y_Z == True:
                    angle_y_Z_i = np.arccos(
                        np.dot(y_XY_ref_IMU, y_XY_i) / (np.linalg.norm(y_XY_ref_IMU) * np.linalg.norm(y_XY_i)))
                    IMU_angle_y_Z.append(angle_y_Z_i * 180 / (np.pi))
                if find_angle_IMU_z_Z == True:
                    angle_z_Z_i = np.arccos(
                        np.dot(z_XY_ref_IMU, z_XY_i) / (np.linalg.norm(z_XY_ref_IMU) * np.linalg.norm(z_XY_i)))
                    IMU_angle_z_Z.append(angle_z_Z_i * 180 / (np.pi))

        # From Clust data
        for row in range(len(Clust_df)):
            # Don't calculate it for first value since vector 1 = vector 2 and arccos function can't solve
            if row == 0:
                Clust_angle_x_Z.append(0)
                Clust_angle_y_Z.append(0)
                Clust_angle_z_Z.append(0)
            else:
                Clust_i_R = R.from_quat(list(
                    [Clust_df.values[row, 1], Clust_df.values[row, 2], Clust_df.values[row, 3], Clust_df.values[row, 0]]))
                Clust_i = Clust_i_R.as_matrix()
                x_XY_i = np.array((Clust_i[0, 0], Clust_i[1, 0]))
                y_XY_i = np.array((Clust_i[0, 1], Clust_i[1, 1]))
                z_XY_i = np.array((Clust_i[0, 2], Clust_i[1, 2]))
                # Apply equation: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
                if find_angle_Clust_x_Z == True:
                    angle_x_Z_i = np.arccos(
                        np.dot(x_XY_ref_Clust, x_XY_i) / (np.linalg.norm(x_XY_ref_Clust) * np.linalg.norm(x_XY_i)))
                    Clust_angle_x_Z.append(angle_x_Z_i * 180 / (np.pi))
                if find_angle_Clust_y_Z == True:
                    angle_y_Z_i = np.arccos(
                        np.dot(y_XY_ref_Clust, y_XY_i) / (np.linalg.norm(y_XY_ref_Clust) * np.linalg.norm(y_XY_i)))
                    Clust_angle_y_Z.append(angle_y_Z_i * 180 / (np.pi))
                if find_angle_Clust_z_Z == True:
                    angle_z_Z_i = np.arccos(
                        np.dot(z_XY_ref_Clust, z_XY_i) / (np.linalg.norm(z_XY_ref_Clust) * np.linalg.norm(z_XY_i)))
                    Clust_angle_z_Z.append(angle_z_Z_i * 180 / (np.pi))

    ### CALCULATE THE DIFFERENCE BETWEEN IMU AND CLUSTER

    # Calculate the absolute difference in angle between cluster and IMU.
    diff_list = []
    IMU_1 = IMU_angle_x_Y
    Clust_1 = Clust_angle_x_Y
    IMU_choice = "IMU_angle_x_Y"
    Clust_choice = "Clust_angle_x_Y"
    for i in range(len(IMU_angle_x_Y)):
        diff_i = abs(IMU_1[i] - Clust_1[i])
        diff_list.append(diff_i)

    # Calculate average difference
    diff = pd.DataFrame(diff_list)
    diff_median = round(diff[0].median(), 3)

    # Save the data from this rep in the log
    logging.info("Local Axes Choice: " + IMU_choice + " " + Clust_choice)
    logging.info("Vector angle difference median: " + str(diff_median))
    # logging.info("Angular velocity median: " + str(median_ang_vel))
    # logging.info("Angular velocity 25th, 75th, 99th percentile: " + str(q25) + ", " + str(q75) + ", " + str(q99))

    # Output data to a csv file
    # diff_array = np.array(diff_list)
    # np.savetxt(output_file_name, diff_array, delimiter=',', fmt='%.6f', header=tag+degrees, comments='')


    ### PLOT THE VECTOR ANGLE RESULTS

    # Boxplot the difference values
    plt.figure(1)
    fig = plt.figure(figsize=(10, 8))
    bp = sns.boxplot(data=diff, notch='True')
    bp.set_xticklabels(["Median = " + str(diff_median)])
    plt.title("Difference between IMU x local axis and Cluster y local axis around " + str(glob_axis))
    plt.ylabel('Degrees')
    plt.savefig("Boxplot_Vector_Angle_Error.png")
    plt.clf()

    # Plot the angles against time
    if glob_axis == "Y":
        ## Plot the change in angle of each local axis, as well as difference chosen above
        plt.figure(2)
        fig = plt.figure(figsize=(10, 8))
        x = time
        y1 = IMU_angle_x_Y
        y2 = IMU_angle_y_Y
        y3 = IMU_angle_z_Y
        y4 = Clust_angle_x_Y
        y5 = Clust_angle_y_Y
        y6 = Clust_angle_z_Y
        y7 = diff
        if find_angle_IMU_x_Y == True:
            plt.scatter(x, y1, s=3)
        if find_angle_IMU_y_Y == True:
            plt.scatter(x, y2, s=3)
        if find_angle_IMU_z_Y == True:
            plt.scatter(x, y3, s=3)
        if find_angle_Clust_x_Y == True:
            plt.scatter(x, y4, s=3)
        if find_angle_Clust_y_Y == True:
            plt.scatter(x, y5, s=3)
        if find_angle_Clust_z_Y == True:
            plt.scatter(x, y6, s=3)
        plt.scatter(x, y7, s=3)
        plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
        plt.title("Rotation of Local Axes around Global Axis: " + glob_axis)
        plt.xlabel('Time')
        plt.ylabel('Degrees')
        plt.legend(["IMU Local X", "Cluster Local X", "Difference in X"])
        # plt.show()
        plt.savefig(vec_angle_output_plot_file)
        plt.clf()

    if glob_axis == "X":
        ## Plot the change in angle of each local axis, as well as difference chosen above
        plt.figure(1)
        fig = plt.figure(figsize=(10, 8))
        x = time
        y1 = IMU_angle_x_X
        y2 = IMU_angle_y_X
        y3 = IMU_angle_z_X
        y4 = Clust_angle_x_X
        y5 = Clust_angle_y_X
        y6 = Clust_angle_z_X
        y7 = diff
        if find_angle_IMU_x_X == True:
            plt.scatter(x, y1, s=3)
        if find_angle_IMU_y_X == True:
            plt.scatter(x, y2, s=3)
        if find_angle_IMU_z_X == True:
            plt.scatter(x, y3, s=3)
        if find_angle_Clust_x_X == True:
            plt.scatter(x, y4, s=3)
        if find_angle_Clust_y_X == True:
            plt.scatter(x, y5, s=3)
        if find_angle_Clust_z_Z == True:
            plt.scatter(x, y6, s=3)
        plt.scatter(x, y7, s=3)
        plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
        plt.title("Rotation of Local Axes around Global Axis: " + glob_axis)
        plt.xlabel('Time')
        plt.ylabel('Degrees')
        plt.legend(["IMU Local Y", "Cluster Local Y", "Difference in Y"], loc="lower right")
        # plt.show()
        plt.savefig(vec_angle_output_plot_file)
        plt.clf()

    if glob_axis == "Z":
        ## Plot the change in angle of each local axis, as well as difference chosen above
        plt.figure(1)
        fig = plt.figure(figsize=(10, 8))
        x = time
        y1 = IMU_angle_x_Z
        y2 = IMU_angle_y_Z
        y3 = IMU_angle_z_Z
        y4 = Clust_angle_x_Z
        y5 = Clust_angle_y_Z
        y6 = Clust_angle_z_Z
        y7 = diff
        if find_angle_IMU_x_z == True:
            plt.scatter(x, y1, s=3)
        if find_angle_IMU_y_Z == True:
            plt.scatter(x, y2, s=3)
        if find_angle_IMU_z_Z == True:
            plt.scatter(x, y3, s=3)
        if find_angle_Clust_x_Z == True:
            plt.scatter(x, y4, s=3)
        if find_angle_Clust_y_Z == True:
            plt.scatter(x, y5, s=3)
        if find_angle_Clust_z_Z == True:
            plt.scatter(x, y6, s=3)
        plt.scatter(x, y7, s=3)
        plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
        plt.title("Rotation of Local Axes around Global Axis: " + glob_axis)
        plt.xlabel('Time')
        plt.ylabel('Degrees')
        plt.legend(["IMU Local X", "Cluster Local X", "Difference in X"], loc="lower right")
        # plt.show()
        plt.savefig(vec_angle_output_plot_file)
        plt.clf()

### CALCULATE AND PLOT EULER ANGLES

if plot_eulers == True:

    # Calculate the angle between quats at every time sample
    N = len(IMU_df)
    IMU_angle1 = np.zeros((N))
    IMU_angle2 = np.zeros((N))
    IMU_angle3 = np.zeros((N))
    Clust_angle1 = np.zeros((N))
    Clust_angle2 = np.zeros((N))
    Clust_angle3 = np.zeros((N))

    for row in range(len(IMU_df)):
        quat_IMU = IMU_df.values[row]
        quat_Clust = Clust_df.values[row]
        quat_IMU_R = R.from_quat([quat_IMU[1], quat_IMU[2], quat_IMU[3], quat_IMU[0]])
        quat_Clust_R = R.from_quat([quat_Clust[1], quat_Clust[2], quat_Clust[3], quat_Clust[0]])
        eul_IMU = quat_IMU_R.as_euler(decomp_seq, degrees=True)
        eul_Clust = quat_Clust_R.as_euler(decomp_seq, degrees=True)
        # if 50 < eul_IMU[0] or 50 < eul_Clust[0] or -100 < eul_IMU[1] < -80 or -100 < eul_Clust[1] < -80:
        #     IMU_angle1[row] = 0
        #     IMU_angle2[row] = 0
        #     IMU_angle3[row] = 0
        #     Clust_angle1[row] = 0
        #     Clust_angle2[row] = 0
        #     Clust_angle3[row] = 0
        # else:
        IMU_angle1[row] = eul_IMU[0]
        IMU_angle2[row] = eul_IMU[1]
        IMU_angle3[row] = eul_IMU[2]
        Clust_angle1[row] = eul_Clust[0]
        Clust_angle2[row] = eul_Clust[1]
        Clust_angle3[row] = eul_Clust[2]

    ### Calculate differences

    diff_angle_1 = np.zeros((N))
    diff_angle_2 = np.zeros((N))
    diff_angle_3 = np.zeros((N))

    for row in range(len(IMU_df)):
        diff_angle_1[row] = abs(IMU_angle1[row] - Clust_angle1[row])
        diff_angle_2[row] = abs(IMU_angle2[row] - Clust_angle2[row])
        diff_angle_3[row] = abs(IMU_angle3[row] - Clust_angle3[row])

    # Calculate average differences
    euler_diff_median_1 = round(np.median(diff_angle_1), 3)
    euler_diff_median_2 = round(np.median(diff_angle_2), 3)
    euler_diff_median_3 = round(np.median(diff_angle_3), 3)
    logging.info("Average difference in Euler Angle 1: " + str(euler_diff_median_1))
    logging.info("Average difference in Euler Angle 2: " + str(euler_diff_median_2))
    logging.info("Average difference in Euler Angle 3: " + str(euler_diff_median_3))

    all_eul_diff_data = np.column_stack((diff_angle_1, diff_angle_2, diff_angle_3))
    output_file_header = "Diff_1, Diff_2, Diff_3"
    np.savetxt("Eul_diff_" + tag + ".csv", all_eul_diff_data, delimiter=',', fmt='%.6f', header=output_file_header, comments="")

    # Plot the euler angles of both IMU and Cluster
    plt.figure(1)
    fig = plt.figure(figsize=(10, 8))
    x = time
    y1 = IMU_angle1
    y2 = IMU_angle2
    y3 = IMU_angle3
    y4 = Clust_angle1
    y5 = Clust_angle2
    y6 = Clust_angle3
    y7 = diff_angle_1
    y8 = diff_angle_2
    y9 = diff_angle_3
    plt.scatter(x, y1, s=1, c='orange')
    plt.scatter(x, y2, s=1, c='blue')
    plt.scatter(x, y3, s=1, c='limegreen')
    plt.scatter(x, y4, s=1, c='red')
    plt.scatter(x, y5, s=1, c='cornflowerblue')
    plt.scatter(x, y6, s=1, c='darkgreen')
    # plt.scatter(x, y7, s=1, c='indianred')
    # plt.scatter(x, y8, s=1, c='midnightblue')
    # plt.scatter(x, y9, s=1, c='olivedrab')
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    plt.title("Euler Angles - Decomp Sequence: " + decomp_seq)
    plt.xlabel('Time')
    plt.ylabel('Degrees')
    # plt.grid(axis="x", which="both")
    # x_range = round(len(time)/sample_rate)
    # plt.xticks(range(0, x_range, 1))
    plt.legend(["IMU Angle 1", "IMU Angle 2", "IMU Angle 3", "Cluster Angle 1", "Cluster Angle 2", "Cluster Angle 3"], loc="lower left", markerscale=3)
    # plt.show()
    plt.savefig("Euler_Angle_Plot_" + tag + ".png")
    plt.clf()

### CALCUALTE QUATENRION DIFFERENCE BETWEEN IMU AND CLUSTER AND BREAK DOWN INTO EULER ANGLES

if plot_quat_difference == True:

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
    mean_angle_1 = np.mean(abs_diff_angle_1)
    mean_angle_2 = np.mean(abs_diff_angle_2)
    mean_angle_3 = np.mean(abs_diff_angle_3)
    sd_angle_1 = np.std(abs_diff_angle_1)
    sd_angle_2 = np.std(abs_diff_angle_2)
    sd_angle_3 = np.std(abs_diff_angle_3)

    logging.info("Quaternion Difference Results:")
    logging.info("Angle 1 mean: " + str(mean_angle_1))
    logging.info("Angle 2 mean: " + str(mean_angle_2))
    logging.info("Angle 3 mean: " + str(mean_angle_3))
    logging.info("Angle 1 SD: " + str(sd_angle_1))
    logging.info("Angle 2 SD: " + str(sd_angle_2))
    logging.info("Angle 3 SD: " + str(sd_angle_3))

    all_quat_diff_data = np.column_stack((abs_diff_angle_1, abs_diff_angle_2, abs_diff_angle_3))
    output_file_header = "Diff_1, Diff_2, Diff_3"
    np.savetxt("Quat_diff_" + tag + ".csv", all_quat_diff_data, delimiter=',', fmt='%.6f', header=output_file_header, comments="")

### CALCUALTE SMALLEST ANGLE BETWEEN QUATERNIONS

if plot_smallest_angle == True:

    angle = np.zeros((N))
    for row in range(len(IMU_df)):
        quat_mult = quaternion_multiply(IMU_df.values[row], quaternion_conjugate(Clust_df.values[row]))
        norm_vec_part = np.linalg.norm([quat_mult[1], quat_mult[2], quat_mult[3]])
        angle[row] = (2 * np.arcsin(norm_vec_part)) * 180 / np.pi

    mean_angle = np.nanmean(angle)
    sd_angle = np.nanstd(angle)
    logging.info("Average Smallest Angle (mean): " + str(mean_angle))
    logging.info("Average Smallest Angle (SD): " + str(sd_angle))

    np.savetxt("Smallest_angle_" + tag + ".csv", angle, delimiter=',', fmt='%.6f', header=str(tag), comments='')