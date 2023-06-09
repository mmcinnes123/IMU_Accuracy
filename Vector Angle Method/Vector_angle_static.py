# This script takes a .csv file with the transformed data, calculates angles between local axis and global,
# then outputs a .csv file with the resultant differences between IMU and Cluster
# This script is specific to reading static data, with gaps in between

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import seaborn as sns
import logging

### SETTINGS

input_file = "H_ST_R2_transformed.csv"
degrees = "all"
# Time interval of interest:
start_time = 0
end_time = 42
# Which cluster quats? OptiTrack or New_ax
Which_clust = "OptiTrack"
# Global Axis of interest:
glob_axis = "Y"
# Choose which angles you want to calculate and plot:
find_angle_IMU_x_Y = True
find_angle_IMU_y_Y = True
find_angle_IMU_z_Y = True
find_angle_IMU_x_X = True
find_angle_IMU_y_X = True
find_angle_IMU_z_X = True
find_angle_IMU_x_Z = True
find_angle_IMU_y_Z = True
find_angle_IMU_z_Z = True
find_angle_Clust_x_Y = True
find_angle_Clust_y_Y = True
find_angle_Clust_z_Y = True
find_angle_Clust_x_X = True
find_angle_Clust_y_X = True
find_angle_Clust_z_X = True
find_angle_Clust_x_Z = True
find_angle_Clust_y_z = True
find_angle_Clust_z_Z = True
plot_diff = True
# Chose which values to compare (line 300)
# Add nrow = num_rows into data readers if times matter


# Set up
sample_rate = 100
skip_rows = start_time*sample_rate
num_rows = (end_time-start_time)*sample_rate

tag = input_file.replace("_transformed.csv", "")
output_file_name = tag + "_" + degrees + "_" + "diff.csv"
output_plot_file = "Changing_angles_" + tag + ".png"

logging.basicConfig(filename='Vector_angle.log', level=logging.INFO)
logging.info("Input file: " + input_file)
logging.info("Static Position (degrees): " + degrees)
logging.info("Global axis: " + glob_axis)


## Open files and create data frame

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

# IMU data
with open(input_file, 'r') as file:
    columns_IMU = [0, 1, 2, 3]
    IMU_df = pd.read_csv(file, usecols=columns_IMU, skiprows=skip_rows, nrows=num_rows)

# Cluster data
with open(input_file, 'r') as file:
    if Which_clust == "OptiTrack":
        columns_Clust = [4, 5, 6, 7]
    elif Which_clust == "New_ax":
        columns_Clust = [8, 9, 10, 11]
    Clust_df = pd.read_csv(file, usecols=columns_Clust, skiprows=skip_rows, nrows=num_rows)


# Make a time column
time = list(np.arange(0, len(IMU_df)/sample_rate, 1/sample_rate))

### CREATE REFERENCE VECTORS

## From IMU Data

# Create a reference orientation matrices from the first time sample (from which all other vectors will be compared)
IMU_ref_R = R.from_quat(list([IMU_ref_df.values[0, 1], IMU_ref_df.values[0, 2], IMU_ref_df.values[0, 3], IMU_ref_df.values[0, 0]]))
IMU_ref = np.array(IMU_ref_R.as_matrix())

# Create reference vectors for each local axis, and for each global axis of interest
# For rotation around global Y (looking at X and Z elements)
x_XZ_ref_IMU = np.array((IMU_ref[0,0], IMU_ref[2,0]))
y_XZ_ref_IMU = np.array((IMU_ref[0,1], IMU_ref[2,1]))
z_XZ_ref_IMU = np.array((IMU_ref[0,2], IMU_ref[2,2]))
# For rotation around global X (looking at Y and Z elements)
x_YZ_ref_IMU = np.array((IMU_ref[1,0], IMU_ref[2,0]))
y_YZ_ref_IMU = np.array((IMU_ref[1,1], IMU_ref[2,1]))
z_YZ_ref_IMU = np.array((IMU_ref[1,2], IMU_ref[2,2]))
# For rotation around global Z (looking at X and Y elements)
x_XY_ref_IMU = np.array((IMU_ref[0,0], IMU_ref[1,0]))
y_XY_ref_IMU = np.array((IMU_ref[0,1], IMU_ref[1,1]))
z_XY_ref_IMU = np.array((IMU_ref[0,2], IMU_ref[1,2]))

IMU_angle_x_Y = []
IMU_angle_y_Y = []
IMU_angle_z_Y = []
IMU_angle_x_X = []
IMU_angle_y_X = []
IMU_angle_z_X = []
IMU_angle_x_Z = []
IMU_angle_y_Z = []
IMU_angle_z_Z = []

## From Clust Data

# Create a reference orientation matrices from the first time sample (from which all other vectors will be compared)
Clust_ref_R = R.from_quat(list([Clust_ref_df.values[0, 1], Clust_ref_df.values[0, 2], Clust_ref_df.values[0, 3], Clust_ref_df.values[0, 0]]))
Clust_ref = np.array(Clust_ref_R.as_matrix())

# Create reference vectors for each local axis, and for each global axis of interest
# For rotation around global Y (looking at X and Z elements)
x_XZ_ref_Clust = np.array((Clust_ref[0,0], Clust_ref[2,0]))
y_XZ_ref_Clust = np.array((Clust_ref[0,1], Clust_ref[2,1]))
z_XZ_ref_Clust = np.array((Clust_ref[0,2], Clust_ref[2,2]))
# For rotation around global X (looking at Y and Z elements)
x_YZ_ref_Clust = np.array((Clust_ref[1,0], Clust_ref[2,0]))
y_YZ_ref_Clust = np.array((Clust_ref[1,1], Clust_ref[2,1]))
z_YZ_ref_Clust = np.array((Clust_ref[1,2], Clust_ref[2,2]))
# For rotation around global Z (looking at X and Y elements)
x_XY_ref_Clust = np.array((Clust_ref[0,0], Clust_ref[1,0]))
y_XY_ref_Clust = np.array((Clust_ref[0,1], Clust_ref[1,1]))
z_XY_ref_Clust = np.array((Clust_ref[0,2], Clust_ref[1,2]))


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

if glob_axis=="Y":
    # From IMU data
    for row in range(len(IMU_df)):
        # Don't calculate it for first value since vector 1 = vector 2 and arccos function can't solve
        if row == 0:
            IMU_angle_x_Y.append(0)
            IMU_angle_y_Y.append(0)
            IMU_angle_z_Y.append(0)
        else:
            IMU_i_R = R.from_quat(list([IMU_df.values[row, 1], IMU_df.values[row, 2], IMU_df.values[row, 3], IMU_df.values[row, 0]]))
            IMU_i = IMU_i_R.as_matrix()
            x_XZ_i = np.array((IMU_i[0, 0], IMU_i[2, 0]))
            y_XZ_i = np.array((IMU_i[0, 1], IMU_i[2, 1]))
            z_XZ_i = np.array((IMU_i[0, 2], IMU_i[2, 2]))
            # Apply quation: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
            if find_angle_IMU_x_Y==True:
                angle_x_Y_i = np.arccos(np.dot(x_XZ_ref_IMU, x_XZ_i) / (np.linalg.norm(x_XZ_ref_IMU) * np.linalg.norm(x_XZ_i)))
                IMU_angle_x_Y.append(angle_x_Y_i * 180 /(np.pi))
            if find_angle_IMU_y_Y==True:
                angle_y_Y_i = np.arccos(np.dot(y_XZ_ref_IMU, y_XZ_i)/(np.linalg.norm(y_XZ_ref_IMU)*np.linalg.norm(y_XZ_i)))
                IMU_angle_y_Y.append(angle_y_Y_i * 180 /(np.pi))
            if find_angle_IMU_z_Y==True:
                angle_z_Y_i = np.arccos(np.dot(z_XZ_ref_IMU, z_XZ_i)/(np.linalg.norm(z_XZ_ref_IMU)*np.linalg.norm(z_XZ_i)))
                IMU_angle_z_Y.append(angle_z_Y_i * 180 /(np.pi))

    # From Clust data
    for row in range(len(Clust_df)):
        # Don't calculate it for first value since vector 1 = vector 2 and arccos function can't solve
        if row == 0:
            Clust_angle_x_Y.append(0)
            Clust_angle_y_Y.append(0)
            Clust_angle_z_Y.append(0)
        else:
            Clust_i_R = R.from_quat(list([Clust_df.values[row, 1], Clust_df.values[row, 2], Clust_df.values[row, 3], Clust_df.values[row, 0]]))
            Clust_i = Clust_i_R.as_matrix()
            x_XZ_i = np.array((Clust_i[0, 0], Clust_i[2, 0]))
            y_XZ_i = np.array((Clust_i[0, 1], Clust_i[2, 1]))
            z_XZ_i = np.array((Clust_i[0, 2], Clust_i[2, 2]))
            # Apply quation: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
            if find_angle_Clust_x_Y==True:
                angle_x_Y_i = np.arccos(np.dot(x_XZ_ref_Clust, x_XZ_i) / (np.linalg.norm(x_XZ_ref_Clust) * np.linalg.norm(x_XZ_i)))
                Clust_angle_x_Y.append(angle_x_Y_i * 180 / (np.pi))
            if find_angle_Clust_y_Y==True:
                angle_y_Y_i = np.arccos(np.dot(y_XZ_ref_Clust, y_XZ_i)/(np.linalg.norm(y_XZ_ref_Clust)*np.linalg.norm(y_XZ_i)))
                Clust_angle_y_Y.append(angle_y_Y_i * 180 / (np.pi))
            if find_angle_Clust_z_Y==True:
                angle_z_Y_i = np.arccos(np.dot(z_XZ_ref_Clust, z_XZ_i)/(np.linalg.norm(z_XZ_ref_Clust)*np.linalg.norm(z_XZ_i)))
                Clust_angle_z_Y.append(angle_z_Y_i*180/(np.pi))

if glob_axis=="X":

    # From IMU data
    for row in range(len(IMU_df)):
        # Don't calculate it for first value since vector 1 = vector 2 and arccos function can't solve
        if row == 0:
            IMU_angle_x_X.append(0)
            IMU_angle_y_X.append(0)
            IMU_angle_z_X.append(0)
        else:
            IMU_i_R = R.from_quat(list([IMU_df.values[row, 1], IMU_df.values[row, 2], IMU_df.values[row, 3], IMU_df.values[row, 0]]))
            IMU_i = IMU_i_R.as_matrix()
            x_YZ_i = np.array((IMU_i[1, 0], IMU_i[2, 0]))
            y_YZ_i = np.array((IMU_i[1, 1], IMU_i[2, 1]))
            z_YZ_i = np.array((IMU_i[1, 2], IMU_i[2, 2]))
            # Apply equation: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
            if find_angle_IMU_x_X == True:
                angle_x_X_i = np.arccos(np.dot(x_YZ_ref_IMU, x_YZ_i) / (np.linalg.norm(x_YZ_ref_IMU) * np.linalg.norm(x_YZ_i)))
                IMU_angle_x_X.append(angle_x_X_i * 180 / (np.pi))
            if find_angle_IMU_y_X == True:
                angle_y_X_i = np.arccos(np.dot(y_YZ_ref_IMU, y_YZ_i) / (np.linalg.norm(y_YZ_ref_IMU) * np.linalg.norm(y_YZ_i)))
                IMU_angle_y_X.append(angle_y_X_i * 180 / (np.pi))
            if find_angle_IMU_z_X == True:
                angle_z_X_i = np.arccos(np.dot(z_YZ_ref_IMU, z_YZ_i) / (np.linalg.norm(z_YZ_ref_IMU) * np.linalg.norm(z_YZ_i)))
                IMU_angle_z_X.append(angle_z_X_i*180/(np.pi))

    # From Clust data
    for row in range(len(Clust_df)):
        # Don't calculate it for first value since vector 1 = vector 2 and arccos function can't solve
        if row == 0:
            Clust_angle_x_X.append(0)
            Clust_angle_y_X.append(0)
            Clust_angle_z_X.append(0)
        else:
            Clust_i_R = R.from_quat(list([Clust_df.values[row, 1], Clust_df.values[row, 2], Clust_df.values[row, 3], Clust_df.values[row, 0]]))
            Clust_i = Clust_i_R.as_matrix()
            x_YZ_i = np.array((Clust_i[1, 0], Clust_i[2, 0]))
            y_YZ_i = np.array((Clust_i[1, 1], Clust_i[2, 1]))
            z_YZ_i = np.array((Clust_i[1, 2], Clust_i[2, 2]))
            # Apply equation: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
            if find_angle_Clust_x_X == True:
                angle_x_X_i = np.arccos(np.dot(x_YZ_ref_Clust, x_YZ_i) / (np.linalg.norm(x_YZ_ref_Clust) * np.linalg.norm(x_YZ_i)))
                Clust_angle_x_X.append(angle_x_X_i * 180 / (np.pi))
            if find_angle_Clust_y_X == True:
                angle_y_X_i = np.arccos(np.dot(y_YZ_ref_Clust, y_YZ_i) / (np.linalg.norm(y_YZ_ref_Clust) * np.linalg.norm(y_YZ_i)))
                Clust_angle_y_X.append(angle_y_X_i * 180 / (np.pi))
            if find_angle_Clust_z_X == True:
                angle_z_X_i = np.arccos(np.dot(z_YZ_ref_Clust, z_YZ_i) / (np.linalg.norm(z_YZ_ref_Clust) * np.linalg.norm(z_YZ_i)))
                Clust_angle_z_X.append(angle_z_X_i*180/(np.pi))


if glob_axis=="Z":

    # From IMU data
    for row in range(len(IMU_df)):
        # Don't calculate it for first value since vector 1 = vector 2 and arccos function can't solve
        if row == 0:
            IMU_angle_x_Z.append(0)
            IMU_angle_y_Z.append(0)
            IMU_angle_z_Z.append(0)
        else:
            IMU_i_R = R.from_quat(list([IMU_df.values[row, 1], IMU_df.values[row, 2], IMU_df.values[row, 3], IMU_df.values[row, 0]]))
            IMU_i = IMU_i_R.as_matrix()
            x_XY_i = np.array((IMU_i[0, 0], IMU_i[1, 0]))
            y_XY_i = np.array((IMU_i[0, 1], IMU_i[1, 1]))
            z_XY_i = np.array((IMU_i[0, 2], IMU_i[1, 2]))
            # Apply equation: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
            if find_angle_IMU_x_Z == True:
                angle_x_Z_i = np.arccos(np.dot(x_XY_ref_IMU, x_XY_i) / (np.linalg.norm(x_XY_ref_IMU) * np.linalg.norm(x_XY_i)))
                IMU_angle_x_Z.append(angle_x_Z_i * 180 / (np.pi))
            if find_angle_IMU_y_Z == True:
                angle_y_Z_i = np.arccos(np.dot(y_XY_ref_IMU, y_XY_i) / (np.linalg.norm(y_XY_ref_IMU) * np.linalg.norm(y_XY_i)))
                IMU_angle_y_Z.append(angle_y_Z_i * 180 / (np.pi))
            if find_angle_IMU_z_Z == True:
                angle_z_Z_i = np.arccos(np.dot(z_XY_ref_IMU, z_XY_i) / (np.linalg.norm(z_XY_ref_IMU) * np.linalg.norm(z_XY_i)))
                IMU_angle_z_Z.append(angle_z_Z_i * 180 / (np.pi))

    # From Clust data
    for row in range(len(Clust_df)):
        # Don't calculate it for first value since vector 1 = vector 2 and arccos function can't solve
        if row == 0:
            Clust_angle_x_Z.append(0)
            Clust_angle_y_Z.append(0)
            Clust_angle_z_Z.append(0)
        else:
            Clust_i_R = R.from_quat(list([Clust_df.values[row, 1], Clust_df.values[row, 2], Clust_df.values[row, 3], Clust_df.values[row, 0]]))
            Clust_i = Clust_i_R.as_matrix()
            x_XY_i = np.array((Clust_i[0, 0], Clust_i[1, 0]))
            y_XY_i = np.array((Clust_i[0, 1], Clust_i[1, 1]))
            z_XY_i = np.array((Clust_i[0, 2], Clust_i[1, 2]))
            # Apply equation: angle = invcos ( vec1 . vec2 / mag(vec1)*mag(vec2)
            if find_angle_Clust_x_Z == True:
                angle_x_Z_i = np.arccos(np.dot(x_XY_ref_Clust, x_XY_i) / (np.linalg.norm(x_XY_ref_Clust) * np.linalg.norm(x_XY_i)))
                Clust_angle_x_Z.append(angle_x_Z_i * 180 / (np.pi))
            if find_angle_Clust_y_Z == True:
                angle_y_Z_i = np.arccos(np.dot(y_XY_ref_Clust, y_XY_i) / (np.linalg.norm(y_XY_ref_Clust) * np.linalg.norm(y_XY_i)))
                Clust_angle_y_Z.append(angle_y_Z_i * 180 / (np.pi))
            if find_angle_Clust_z_Z == True:
                angle_z_Z_i = np.arccos(np.dot(z_XY_ref_Clust, z_XY_i) / (np.linalg.norm(z_XY_ref_Clust) * np.linalg.norm(z_XY_i)))
                Clust_angle_z_Z.append(angle_z_Z_i * 180 / (np.pi))



### CALCULATE THE DIFFERENCE BETWEEN IMU AND CLUSTER

# Calculate the difference in angle between cluster and IMU.
diff_list = []
IMU_1 = IMU_angle_x_Y
Clust_1 = Clust_angle_x_Y
IMU_choice = "IMU_angle_x_Y"
Clust_choice = "Clust_angle_x_Y"
for i in range(len(IMU_angle_x_Y)):
    diff_i = abs(IMU_1[i] - Clust_1[i])
    diff_list.append(diff_i)

# # Calculate the angular velocity between every time sample
# ang_vel = []
# for i in range(len(IMU_1)-1):
#     ang_vel_i = abs((IMU_1[i+1] - IMU_1[i]))*sample_rate
#     # Only include data points from IMU when it's 'moving' (use 20deg/s as cut off)
#     if ang_vel_i >= 20:
#         ang_vel.append(ang_vel_i)
#
# median_ang_vel = np.median(ang_vel)
# q99, q75, q25 = np.percentile(ang_vel, [99, 75 ,25])

# Save the data from this rep in the log
diff = pd.DataFrame(diff_list)
diff_median = round(diff[0].median(), 3)
logging.info("Local Axes Choice: " + IMU_choice + " " + Clust_choice)
logging.info("Difference median: " + str(diff_median))
# logging.info("Angular velocity median: " + str(median_ang_vel))
# logging.info("Angular velocity 25th, 75th, 99th percentile: " + str(q25) + ", " + str(q75) + ", " + str(q99))

# Output data to a csv file
diff_array = np.array(diff_list)
np.savetxt(output_file_name, diff_array, delimiter=',', fmt='%.6f', header=tag+degrees, comments='')



### PLOT THE RESULTS

# Boxplot the difference values

# plt.figure(1)
# fig = plt.figure(figsize=(10, 8))
# bp = sns.boxplot(data=diff, notch='True')
#
# bp.set_xticklabels(["Median = " + str(diff_median)])
# # Adding title
# plt.title("Difference between IMU x local axis and Cluster y local axis around " + str(glob_axis))
# plt.ylabel('Degrees')
# # Save plot
# plt.savefig("Diff.png")
# plt.clf()


# Plot the angles against time

if glob_axis=="Y":
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
    if plot_diff == True:
        plt.scatter(x, y7, s=3)
    plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
    plt.title("Rotation of Local Axes around Global Axis" + glob_axis)
    plt.xlabel('Time')
    plt.ylabel('Degrees')
    plt.legend(["IMU Local X", "IMU Local Y", "IMU Local Z", "Cluster Local X", "Cluster Local Y", "Cluster Local Z", "Difference in X"])
    # plt.legend(["IMU Local X", "IMU Local Y", "IMU Local Z", "Cluster Local X", "Cluster Local Y", "Cluster Local Z", "Difference in X"])
    # plt.show()
    plt.savefig(output_plot_file)
    plt.clf()

if glob_axis=="X":
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
    if plot_diff == True:
        plt.scatter(x, y7, s=3)
    plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
    plt.title("Rotation of Local Axes around Global Axis" + glob_axis)
    plt.xlabel('Time')
    plt.ylabel('Degrees')
    plt.legend(["IMU Local X","IMU Local Y", "IMU Local Z", "Cluster Local X", "Cluster Local Y", "Cluster Local Z", "Difference in X"], loc ="lower right")
    # plt.show()
    plt.savefig(output_plot_file)
    plt.clf()

if glob_axis=="Z":
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
    if plot_diff == True:
        plt.scatter(x, y7, s=3)
    plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
    plt.title("Rotation of Local Axes around Global Axis" + glob_axis)
    plt.xlabel('Time')
    plt.ylabel('Degrees')
    plt.legend(["IMU Local X","IMU Local Y", "IMU Local Z", "Cluster Local X", "Cluster Local Y", "Cluster Local Z", "Difference in X"], loc ="lower right")
    # plt.show()
    plt.savefig(output_plot_file)
    plt.clf()