# Script for quick review of movement - to  plot the quaternions or write to APDM file format to view in OpenSim.

from pre_process import *
from analysis import plot_the_quats

# Make sure APDM file template is in same folder, with numbers as first row,
# and frequency set to match the recording freq (100Hz)


# SETTINGS
file_label = "CON_MP"
APDM_template_file = "APDM_template_4S.csv"
sample_rate = 100
int_decomp_seq = "YXZ"  # Intrinsic decomposition seq - used for plotting euler angles
ext_decomp_seq = "yxz"  # Extrinisic decomposition seq - used for calculating quaternion difference


def full_analysis(input_file):
    tag = input_file.replace(" - Report2.txt", "")
    ### SETTINGS

    # Choose outputs
    plot_quats = True
    write_APDM = False
    write_text_file = False
    plot_IMUvsClust_eulers = False
    plot_fourIMUs_eulers = False
    plot_proj_vector_angle = False
    plot_BA = False
    plot_BA_Eulers = False

    # Choose global axis of interest for vector angle projection
    global_axis = "X"
    # Choose which OMC LCF: "NewAx" "t0" or "average" or "vel" or 'local'
    which_OMC_LCF = "NewAx"

    ### TRANSFORM THE IMU DATA

    # Read data from the file
    IMU1_df_raw, OpTr_Clus_df_raw, NewAx_Clus_df_raw = read_data_frame_from_file(input_file)

    # Interpolate for missing data
    IMU1_df_raw, OpTr_Clus_df_raw, NewAx_Clus_df_raw, total_nans = interpolate_dfs(IMU1_df_raw, OpTr_Clus_df_raw,
                                                                                   NewAx_Clus_df_raw)
    print("Total missing data points (nans): " + str(total_nans))

    # Transform the IMU data
    IMU1_df = transform_IMU_data(IMU1_df_raw)

    ### TRANSFORM THE CLUSTER DATA

    # Rotate the LCF of all cluster data based on quat difference IMU to cluster at t = 0
    OMC_Clust_t0 = trans_clust_t0(OpTr_Clus_df_raw, IMU1_df)
    # Rotate the LCF of all cluster data based on average quat difference IMU to cluster
    OMC_Clust_average = trans_clust_average(OpTr_Clus_df_raw, IMU1_df)


    # Find the rotation quaternion from OptiTrack's cluster LCF to IMU LCF - at t = 0s
    if which_OMC_LCF == "t0":
        OMC_Clus_df = OMC_Clust_t0
    elif which_OMC_LCF == "average":
        OMC_Clus_df = OMC_Clust_average
    elif which_OMC_LCF == "local":
        OMC_Clus_df = NewAx_Clus_df_raw
    else:
        OMC_Clus_df = OpTr_Clus_df_raw

    ### WRITE DATA TO APDM FILE FORMAT

    # Write the transformed IMU data (ONLY 3/4 IMUS) and original cluster data to an APDM file
    if write_APDM == True:
        write_to_APDM(IMU1_df, OMC_Clus_df, OMC_Clust_t0, OMC_Clust_average, APDM_template_file, tag)

    # Plot two sets of quaternions for comparison and checking timings (set to stylus-defined cluster and IMU1)
    if plot_quats == True:
        plot_the_quats(IMU1_df, OMC_Clus_df, tag, sample_rate)

# Plot the quaternions to find start and end times
for i in [1, 2, 3, 4, 5]:
    file_name = file_label + "_R" + str(i) + " - Report2.txt"
    full_analysis(file_name)

