*** Version 1.1.0 update***
Please do not use biometric_fileread.m from previous versions. Use grabmyo_convert_wfdb_to_mat.py instead
*** 

# Large-Sample and Multiday Surface Electromyogram Recordings of Hand Gestures

We provide the EMG signals recordings of 43 participants over 3 different days while performing multiple hand gestures. The signals have been acquired at a sampling rate of 2048 Hz using the EMGUSB2+ device (OT Bioelletronica, Italy). EMG signals have been collected from 16 locations (channels) on the forearm and 12 locations on the wrist. 

The signals are provided as ".dat" and ".hea" files. The files are organized under 3 folders, "Session1", "Session2" and Session3".  Each folder contains folders with the naming "session{i}_participant{j}" where i = {1,2,3} and j = {1,2,3...43}. Each folder contains files with the naming "session{i}_participant{j}_gesture{k}_trial{l}.dat" and (".hea") where k = {1,2,3...17} and l = {1,2,3...7} Sample matlab codes have been provided to: 
1) read all wfdb format files (.dat and .hea) and save them as .mat files
2) use the converted .mat files for standard frequency division based feature extraction
3) save the feature vectors for all the participants in the form of a database (also as .mat files)
These files processing will facilitate the use of machine learning algorithms for applications such as biometrics and gesture recognition. Note that the original signal files can be processed with any feature extraction algorithm.

To run the codes please follow the steps:
1) run grabmyo_convert_wfdb_to_mat.py (Python code) to convert .dat and .hea files to .mat version.
---Requirements: wfdb python package
2) run grabmyo_visualize.py (Python code) to visualize random trials. 
---Requirements: matplotlib, scipy.io, numpy python package
3) run grabmyo_feature_extraction.m (Matlab code) to begin the signal processing and feature extraction steps.
---Requirements: MATLAB
 
Details about the experimental protocol and participants are provided in "MotionSequence.txt","ParticipantInfo_summary.xlsx", "electrodelocation.pdf","GestureList.JPG" and "DeviceInfo.docx"
For more details about the GrabMyo Database, please see https://physionet.org/  

License:
Open Data Commons Attribution License v1.0

Contact:
Dr. Ning Jiang
Director, Waterloo Engineering Bionics Lab, 
University of Waterloo, Waterloo N2L 3G1, Canada.
ning.jiang@uwaterloo.ca
