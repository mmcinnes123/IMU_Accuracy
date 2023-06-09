
print("Running " + "main as " + __name__)

from transform import read_data_frame_from_file
from transform import transform_IMU_data
from transform import write_to_APDM
from transform import write_to_txt
from transform import plot_the_quats
from transform import interpolate_dfs
from find_eulers import eulers_from_quats
from find_eulers import plot_all_eulers

### SETTINGS
# Make sure APDM file template is in same folder, with numbers as first row, and frequency set to match the recording freq (100Hz)
input_file = "Inter_IMU_R1 - Report1.txt"
tag = input_file.replace(" - Report1.txt", "")
template_file = "APDM_template_4S.csv"
sample_rate = 100
decomp_seq = "YZX"

# Choose what outputs you want
write_APDM = False
plot_quats = True
write_text_file = False

### TRANSFORM THE IMU DATA, PLOT THE QUATERNIONS, AND WRITE DATA TO APDM AND TXT FILE

# Read data from the file
IMU1_df, IMU2_df, IMU3_df, OpTr_Clus_df, NewAx_Clus_df = read_data_frame_from_file(input_file)

# Transform the IMU data
IMU1_df_trans = transform_IMU_data(IMU1_df)
IMU2_df_trans = transform_IMU_data(IMU2_df)
IMU3_df_trans = transform_IMU_data(IMU3_df)

# Interpolate for missing data
IMU1_df_trans, IMU2_df_trans, IMU3_df_trans, OpTr_Clus_df, NewAx_Clus_df, total_nans = interpolate_dfs(IMU1_df_trans, IMU2_df_trans, IMU3_df_trans, OpTr_Clus_df, NewAx_Clus_df)


# Write the transformed IMU data and original cluster data to an APDM file
if write_APDM == True:
    write_to_APDM(IMU1_df_trans, IMU2_df_trans, IMU3_df_trans, NewAx_Clus_df, template_file, tag)

# Write the transformed IMU data and original cluster data to a .txt file
if write_text_file == True:
    write_to_txt(IMU1_df_trans, IMU2_df_trans, IMU3_df_trans, OpTr_Clus_df, NewAx_Clus_df, tag)

# Plot two sets of quaternions for comparison and checking timings (set to stylus-defined cluster and IMU1)
if plot_quats == True:
    plot_the_quats(IMU1_df_trans, NewAx_Clus_df, tag, sample_rate)


# Calculate Euler angles from the quaternions
IMU1_eul_1, IMU1_eul_2, IMU1_eul_3 = eulers_from_quats(IMU1_df_trans, decomp_seq)
IMU2_eul_1, IMU2_eul_2, IMU2_eul_3 = eulers_from_quats(IMU2_df_trans, decomp_seq)
IMU3_eul_1, IMU3_eul_2, IMU3_eul_3 = eulers_from_quats(IMU3_df_trans, decomp_seq)
OpTr_Clus_eul_1, OpTr_Clus_eul_2, OpTr_Clus_eul_3 = eulers_from_quats(OpTr_Clus_df, decomp_seq)
NewAx_Clus_eul_1, NewAx_Clus_eul_2, NewAx_Clus_eul_3 = eulers_from_quats(NewAx_Clus_df, decomp_seq)

# Plot the Euler angles
plot_all_eulers(IMU1_eul_1, IMU1_eul_2, IMU1_eul_3, IMU2_eul_1, IMU2_eul_2, IMU2_eul_3, IMU3_eul_1, IMU3_eul_2, IMU3_eul_3, NewAx_Clus_eul_1, NewAx_Clus_eul_2, NewAx_Clus_eul_3, sample_rate, decomp_seq, tag)
