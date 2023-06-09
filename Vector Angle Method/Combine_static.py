
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import seaborn as sns
import logging

tag = "H_ST"

input_file1 = tag + "_R2_90_diff.csv"
input_file2 = tag + "_R2_180_diff.csv"
input_file3 = tag + "_R2_270_diff.csv"
input_file4 = tag + "_R3_90_diff.csv"
input_file5 = tag + "_R3_180_diff.csv"
input_file6 = tag + "_R3_270_diff.csv"
input_file7 = tag + "_R4_90_diff.csv"
input_file8 = tag + "_R4_180_diff.csv"
input_file9 = tag + "_R4_270_diff.csv"
input_file10 = tag + "_R5_90_diff.csv"
input_file11 = tag + "_R5_180_diff.csv"
input_file12 = tag + "_R5_270_diff.csv"

# Open files and create data frames
with open(input_file1, 'r') as file:
    df_rep2_90 = pd.read_csv(file, header=0)
with open(input_file2, 'r') as file:
    df_rep2_180 = pd.read_csv(file, header=0)
with open(input_file3, 'r') as file:
    df_rep2_270 = pd.read_csv(file, header=0)
with open(input_file4, 'r') as file:
    df_rep3_90 = pd.read_csv(file, header=0)
with open(input_file5, 'r') as file:
    df_rep3_180 = pd.read_csv(file, header=0)
with open(input_file6, 'r') as file:
    df_rep3_270 = pd.read_csv(file, header=0)
with open(input_file7, 'r') as file:
    df_rep4_90 = pd.read_csv(file, header=0)
with open(input_file8, 'r') as file:
    df_rep4_180 = pd.read_csv(file, header=0)
with open(input_file9, 'r') as file:
    df_rep4_270 = pd.read_csv(file, header=0)
with open(input_file10, 'r') as file:
    df_rep5_90 = pd.read_csv(file, header=0)
with open(input_file11, 'r') as file:
    df_rep5_180 = pd.read_csv(file, header=0)
with open(input_file12, 'r') as file:
    df_rep5_270 = pd.read_csv(file, header=0)

# Combine all data

rep1 = df_rep2_90.to_numpy()
rep2 = df_rep2_180.to_numpy()
rep3 = df_rep2_270.to_numpy()
rep4 = df_rep3_90.to_numpy()
rep5 = df_rep3_180.to_numpy()
rep6 = df_rep3_270.to_numpy()
rep7 = df_rep4_90.to_numpy()
rep8 = df_rep4_180.to_numpy()
rep9 = df_rep4_270.to_numpy()
rep10 = df_rep5_90.to_numpy()
rep11 = df_rep5_180.to_numpy()
rep12 = df_rep5_270.to_numpy()

all_data = np.concatenate((rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8, rep9, rep10, rep11, rep12), axis=0)



# Calculate median and IQR
all_data_df = pd.DataFrame(all_data)
all_data_median = round(all_data_df[0].median(), 2)
q75, q25 = np.percentile(all_data_df[0], [75 ,25])
iqr = q75 - q25

logging.basicConfig(filename='Combined_data_' + tag + '.log', level=logging.INFO)
logging.info("Median: " + str(all_data_median))
logging.info("IQR: " + str(iqr))

# Boxplot the results

plt.figure(1)
fig = plt.figure(figsize=(10, 8))
bp = sns.boxplot(data=all_data_df, notch='True')
bp.set_xticklabels(["Median = " + str(all_data_median)])
plt.title("Difference between IMU and Cluster over all repetitions of " + tag)
plt.ylabel('Degrees')
plt.savefig("Combined_data_diff_" + tag + ".png")
plt.clf()