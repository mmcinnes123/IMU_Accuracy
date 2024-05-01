# IMU_Accuracy
Using quaternion comparison to find relative accuracy of inertial sensor by comparing with rigidly attached Optical Motion Capture marker cluster data. 

main_a.py – find the global misalignment quaternion

main_b.py – pre-process all CON and FUN data, create csv files for inter and absolute processing

main_c.py – find absolute error for all CON_MP and CON_SP and FUN data

main_d.py – find absolute error for RoM (from CON_HO and CON_VE)

main_e.py – find inter-IMU error for all CON_MP and CON_SP data

main_f.py – find inter_IMU error for RoM (from CON_HO and CON_VE)

main_g.py – find joint angle angle errors for all FUN data

# This code was used to produce the results presented in poster at BioMedEng23
