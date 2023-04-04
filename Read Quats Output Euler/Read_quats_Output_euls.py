import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

### SETTINGS:
input_file = "Raw_data_transposed_andchanged.csv"
IMU_decomp_seq = 'zxy'
Clus_decomp_seq = 'zxy'
# Time interval 1, 2 or 3. 1 - vertical axis, 2 - horizontal axis 1, 3 - horizontal axis 2
time_int = 3
# Make sure time intervals are correct below
# Make sure correct columns have been chosen from the data file

### NOTES:
# For time interval 1:
# decomp seq which works for both: xzy
# For time interval 2:
# decomp seq which works for both: xyz
# For time interval 3:
# decomp seq which works for both: zyx


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
    columns_Clus = [40,41,42,43]
    Clus_df = pd.read_csv(file, usecols=columns_Clus, skiprows=skip_rows, nrows = num_rows)

# Use New Axes defined with the stylus?
# with open(input_file, 'r') as file:
#     columns_Clus = [55,56,57,58]
#     Clus_df = pd.read_csv(file, usecols=columns_Clus, skiprows=skip_rows, nrows = num_rows)

# Make a time column
time = list(range(len(IMU_df)))

# Make empty lists for the Euler angles (note: 1,2,3 is equal to decomposition order, not 'x, y, z')
IMU_1 = []
IMU_2 = []
IMU_3 = []
Clus_1 = []
Clus_2 = []
Clus_3 = []

# For every time sample in the data frame, create a scipy rotation and output euler angles (constrained by the time interval specified)
for row in range(len(IMU_df)):
    IMU_i = list([IMU_df.values[row, 1], IMU_df.values[row, 2], IMU_df.values[row, 3], IMU_df.values[row, 0]])
    quat_i = R.from_quat(IMU_i)
    eul_i = quat_i.as_euler(IMU_decomp_seq, degrees=True)
    alpha_i = eul_i[0]
    beta_i = eul_i[1]
    gamma_i = eul_i[2]
    IMU_1.append(alpha_i)
    IMU_2.append(beta_i)
    IMU_3.append(gamma_i)

for row in range(len(Clus_df)):
    Clus_i = list([Clus_df.values[row, 1], Clus_df.values[row, 2], Clus_df.values[row, 3], Clus_df.values[row, 0]])
    quat_i = R.from_quat(Clus_i)
    eul_i = quat_i.as_euler(Clus_decomp_seq, degrees=True)
    alpha_i = eul_i[0]
    beta_i = eul_i[1]
    gamma_i = eul_i[2]
    Clus_1.append(alpha_i)
    Clus_2.append(beta_i)
    Clus_3.append(gamma_i)


# Plot the euler angles
plt.figure(1)
fig = plt.figure(figsize=(10, 8))
x = time
y1 = IMU_1
y2 = IMU_2
y3 = IMU_3
y4 = Clus_1
y5 = Clus_2
y6 = Clus_3
plt.scatter(x,y1, s = 3)
plt.scatter(x,y2, s = 3)
plt.scatter(x,y3, s = 3)
plt.scatter(x,y4, s = 3)
plt.scatter(x,y5, s = 3)
plt.scatter(x,y6, s = 3)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title("Comparison of Euler Angles " + "IMU seq: (1,2,3)" + IMU_decomp_seq + " Cluster seq: (1,2,3)" + Clus_decomp_seq)
plt.xlabel('Time')
plt.ylabel('Degrees')
plt.legend(["IMU 1", "IMU 2", "IMU 3", "Cluster 1", "Cluster 2", "Cluster 3"])
# plt.show()
plt.savefig("Time_int_" + str(time_int) + "_IMU_seq_" + IMU_decomp_seq + "Clus_seq_" + Clus_decomp_seq + ".png")
# plt.clf()clf