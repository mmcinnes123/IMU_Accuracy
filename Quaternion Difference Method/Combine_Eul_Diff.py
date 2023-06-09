
import pandas as pd
from scipy.spatial.transform import Rotation as R
import seaborn as sns
import logging
import matplotlib.pyplot as plt
import numpy as np

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = abs(data1 - data2)                  # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    CI_low    = md - 1.96*sd
    CI_high   = md + 1.96*sd
    # Calculate how many data points are within the cofidence interval
    count = 0
    for i in range(len(diff)):
        if CI_low <= diff[i] <= CI_high:
            count = count + 1
    percent_within_LoA = round((100 * count / len(diff)), 1)

    plt.scatter(mean, diff, *args, **kwargs, s=5)
    plt.axhline(md,           color='black', linestyle='-')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    return md, sd, mean, CI_low, CI_high, diff, percent_within_LoA

tag = "MP_H"
decomp_seq = "XYZ"

input_file1 = "Eul_diff_" + tag + "_R1.csv"
input_file2 = "Eul_diff_" + tag + "_R2.csv"
input_file3 = "Eul_diff_" + tag + "_R3.csv"
input_file4 = "Eul_diff_" + tag + "_R4.csv"
input_file5 = "Eul_diff_" + tag + "_R5.csv"



# Open files and create data frames
with open(input_file1, 'r') as file:
    df_rep1 = pd.read_csv(file, header=0)
with open(input_file2, 'r') as file:
    df_rep2 = pd.read_csv(file, header=0)
with open(input_file3, 'r') as file:
    df_rep3 = pd.read_csv(file, header=0)
with open(input_file4, 'r') as file:
    df_rep4 = pd.read_csv(file, header=0)
with open(input_file5, 'r') as file:
    df_rep5 = pd.read_csv(file, header=0)

### COMBINE ALL DATA

# Add together all the angles around the 1st decomp axis
rep1_1 = df_rep1["Diff_1"].to_numpy()
rep2_1 = df_rep2["Diff_1"].to_numpy()
rep3_1 = df_rep3["Diff_1"].to_numpy()
rep4_1 = df_rep4["Diff_1"].to_numpy()
rep5_1 = df_rep5["Diff_1"].to_numpy()
all_data_1 = np.concatenate((rep1_1, rep2_1, rep3_1, rep4_1, rep5_1), axis=0)

# Add together all the angles around the 2nd decomp axis
rep1_2 = df_rep1[" Diff_2"].to_numpy()
rep2_2 = df_rep2[" Diff_2"].to_numpy()
rep3_2 = df_rep3[" Diff_2"].to_numpy()
rep4_2 = df_rep4[" Diff_2"].to_numpy()
rep5_2 = df_rep5[" Diff_2"].to_numpy()
all_data_2 = np.concatenate((rep1_2, rep2_2, rep3_2, rep4_2, rep5_2), axis=0)

# Add together all the angles around the 3rd decomp axis
rep1_3 = df_rep1[" Diff_3"].to_numpy()
rep2_3 = df_rep2[" Diff_3"].to_numpy()
rep3_3 = df_rep3[" Diff_3"].to_numpy()
rep4_3 = df_rep4[" Diff_3"].to_numpy()
rep5_3 = df_rep5[" Diff_3"].to_numpy()
all_data_3 = np.concatenate((rep1_3, rep2_3, rep3_3, rep4_3, rep5_3), axis=0)


### CALCULATE AVERAGES

all_data_1_df = pd.DataFrame(all_data_1)
all_data_1_no_nans = all_data_1_df.dropna()
# shapiro_results_1 = shapiro(all_data_1_no_nans)
all_data_1_mean = np.nanmean(all_data_1_df)
all_data_1_sd = np.nanstd(all_data_1_df)

all_data_2_df = pd.DataFrame(all_data_2)
all_data_2_no_nans = all_data_2_df.dropna()
# shapiro_results_2 = shapiro(all_data_2_no_nans)
all_data_2_mean = np.nanmean(all_data_2_df)
all_data_2_sd = np.nanstd(all_data_2_df)

all_data_3_df = pd.DataFrame(all_data_3)
all_data_3_no_nans = all_data_3_df.dropna()
# shapiro_results_3 = shapiro(all_data_3_no_nans)
all_data_3_mean = np.nanmean(all_data_3_df)
all_data_3_sd = np.nanstd(all_data_3_df)


logging.basicConfig(filename='Combined_data_' + tag + "_" + decomp_seq + '.log', level=logging.INFO)
logging.info("Decomp Sequence: " + decomp_seq)
logging.info("Eul Diff Mean 1: " + str(all_data_1_mean))
logging.info("Eul Diff SD 1: " + str(all_data_1_sd))
logging.info("Eul Diff Mean 2: " + str(all_data_2_mean))
logging.info("Eul Diff SD 2: " + str(all_data_2_sd))
logging.info("Eul Diff Mean 3: " + str(all_data_3_mean))
logging.info("Eul Diff SD 3: " + str(all_data_3_sd))
