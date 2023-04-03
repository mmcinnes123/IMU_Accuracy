import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

### SETTINGS:
input_file = "Raw_data_for_quat2eul.csv"
IMU_decomp_seq = 'zyx'
Clus_decomp_seq = 'ZXY'

# For time interval 1:
# IMU_decomp_seq = 'zyx'
# Clus_decomp_seq = 'yzx'

# For time interval 2:
# IMU_decomp_seq = 'zyx'
# Clus_decomp_seq = 'ZXY'

# For time interval 3:
# IMU_decomp_seq = 'zyx'
# Clus_decomp_seq = 'ZXY'

# Time interval 1, 2 or 3. 1 - vertical axis, 2 - horizontal axis 1, 3 - horizontal axis 2
time_int = 3

# Time intervals: 1 = 0-19s, 2 = 29-45s, 3 = 52-71s
sample_rate = 74
if time_int ==1:
    num_rows = 19*sample_rate
    skip_rows = range(1,3)
if time_int ==2:
    num_rows = (45-29)*sample_rate
    skip_rows = range(1,((29*sample_rate)+3))
if time_int ==3:
    num_rows = (71-52)*sample_rate
    skip_rows = range(1,((52*sample_rate)+3))

# Open files and create data frame
with open(input_file, 'r') as file:
    columns_IMU = [25,26,27,28]
    IMU_df = pd.read_csv(file, usecols=columns_IMU, skiprows=skip_rows, nrows = num_rows)

with open(input_file, 'r') as file:
    columns_Clus = [55,56,57,58]
    Clus_df = pd.read_csv(file, usecols=columns_Clus, skiprows=skip_rows, nrows = num_rows)


# Make a time column
time = list(range(len(IMU_df)))

# Make empty lists for the Euler angles
IMU_x = []
IMU_y = []
IMU_z = []
Clus_x = []
Clus_y = []
Clus_z = []

for row in range(len(IMU_df)):
    IMU_i = list(IMU_df.values[row,:4])
    quat_i = R.from_quat(IMU_i)
    eul_i = quat_i.as_euler(IMU_decomp_seq, degrees=True)
    x_i = eul_i[0]
    y_i = eul_i[1]
    z_i = eul_i[2]
    IMU_x.append(x_i)
    IMU_y.append(y_i)
    IMU_z.append(z_i)

for row in range(len(Clus_df)):
    Clus_i = list(Clus_df.values[row,:4])
    quat_i = R.from_quat(Clus_i)
    eul_i = quat_i.as_euler(Clus_decomp_seq, degrees=True)
    x_i = eul_i[0]
    y_i = eul_i[1]
    z_i = eul_i[2]
    Clus_x.append(x_i)
    Clus_y.append(y_i)
    Clus_z.append(z_i)


# Plot the euler angles
plt.figure(1)
fig = plt.figure(figsize=(10, 8))
x = time
y1 = IMU_x
y2 = IMU_y
y3 = IMU_z
y4 = Clus_x
y5 = Clus_y
y6 = Clus_z
plt.scatter(x,y1, s = 3)
plt.scatter(x,y2, s = 3)
plt.scatter(x,y3, s = 3)
plt.scatter(x,y4, s = 3)
plt.scatter(x,y5, s = 3)
plt.scatter(x,y6, s = 3)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title("Comparison of Euler Angles " + "IMU seq:" + IMU_decomp_seq + " Cluster seq:" + Clus_decomp_seq)
plt.xlabel('Time')
plt.ylabel('Degrees')
plt.legend(["IMU x", "IMU y", "IMU z", "Cluster x", "Cluster y", "Cluster z"])
# plt.show()
plt.savefig("Time_int_" + str(time_int) + "_IMU_seq_" + IMU_decomp_seq + "Clus_seq_" + Clus_decomp_seq + ".png")
# plt.clf()