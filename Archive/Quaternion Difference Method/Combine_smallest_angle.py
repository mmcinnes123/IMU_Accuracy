
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy.stats import shapiro

tag = "MP_H"
decomp_seq = "XYZ"

input_file1 = "Smallest_angle_" + tag + "_R1.csv"
input_file2 = "Smallest_angle_" + tag + "_R2.csv"
input_file3 = "Smallest_angle_" + tag + "_R3.csv"
input_file4 = "Smallest_angle_" + tag + "_R4.csv"
input_file5 = "Smallest_angle_" + tag + "_R5.csv"

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

# Combine all data

rep1 = df_rep1.to_numpy()
rep2 = df_rep2.to_numpy()
rep3 = df_rep3.to_numpy()
rep4 = df_rep4.to_numpy()
rep5 = df_rep5.to_numpy()

all_data = np.concatenate((rep1, rep2, rep3, rep4, rep5), axis=0)



# Calculate averages

all_data_df = pd.DataFrame(all_data)
all_data_mean = np.nanmean(all_data_df)
all_data_sd = np.nanstd(all_data_df)

all_data_no_nans = all_data_df.dropna()
shapiro_results = shapiro(all_data_no_nans)

logging.basicConfig(filename='Combined_data_' + tag + "_" + decomp_seq + '.log', level=logging.INFO)
logging.info("Smallest Angle Mean: " + str(all_data_mean))
logging.info("Smallest Angle SD: " + str(all_data_sd))
