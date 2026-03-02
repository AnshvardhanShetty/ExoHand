"""
This script will read the .dat and .hea files downloaded locally to
to your hard drive. The downloaded files are organzied in 3 folders
'Session 1','Session 2' and 'Session 3'. Each folder contains 119 data
files (.dat) and 119 header files (.hea).

signal properties
total channels = 28 (16 forearm + 12 wrist)
sampling frequency = 2048 Hz
bandpass filtering (hardware) = 10Hz-500Hz


In order to run this script make sure the above three folders are and
a fileconversion function 'rdwfdb.m' are in the same directory


output %%%%%%%%%
Main Folder: 'Output BM'
Folders: 'Session 1_converted','Session 2_converted', 'Session 3_converted',
Each folder: 43 .mat files
VarOut: DATA_FOREARM, DATA_WRIST (7x17 cell matrices)
DATA_FOREARM: each cell: 5secs*sampfreq x Nchannels numeric array
DATA_WRIST: each cell: 5secs*sampfreq x Nchannels numeric array

Forearm Electrode Configuration %%%%%%%%%
 1  2  3  4  5  6  7  8
 9 10 11 12 13 14 15 16

Wrist Electrode Configuration %%%%%%%%%
 1  2  3  4  5  6
 7  8  9 10 11 12

Written by Ashirbad Pradhan
email: ashirbad.pradhan@uwaterloo.ca
"""

# Your Python code starts here


import sys
import os
import wfdb
from scipy.io import savemat

# Add paths for Session1, Session2, and Session3
session_paths = ['Session1', 'Session2', 'Session3']
for session_path in session_paths:
    sys.path.append(os.path.join(os.getcwd(), session_path))

# Obtain the total number of subjects
nsub = len(
    os.listdir(os.path.join(os.getcwd(), 'Session1'))) - 2  # Assuming number of subjects are the same in all sessions
nsession = 3
ngesture = 16  # Total number of gestures
ntrials = 7  # Total number of trials

# Define output folder
output_folder = 'Output BM'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
else:
    while True:
        print(f"Found existing folder in: {os.getcwd()}")
        cont = input("Overwrite it (Y/N)? ").upper()
        if cont in ('Y', 'N'):
            if cont == 'Y':
                print("Overwriting")
                import shutil

                shutil.rmtree(output_folder)
                os.mkdir(output_folder)
                break
            else:
                print("Exiting Script!")
                sys.exit()

import os
import numpy as np

foldername = []
filename = []
flag = 0
count = 0

# Define channel mappings for forearm and wrist
forearm_channels = np.concatenate((np.ones(8), np.ones(8), np.zeros(8), np.zeros(8)))
wrist_channels = np.concatenate(
    (np.zeros(8), np.zeros(8), np.zeros(1), np.ones(6), np.zeros(2), np.ones(6), np.zeros(1)))
indices = [i for i, x in enumerate(wrist_channels) if x == 1]
print(indices)
# Define data_forearm and data_wrist lists before the loop


# Create a 7x17 array of 2D matrices
matrices_forearm = np.empty((7, 17), dtype=object)
matrices_wrist = np.empty((7, 17), dtype=object)
# Populate each element with a 2D matrix (for demonstration, using zero data)
for i in range(7):
    for j in range(17):
        matrices_forearm[i, j] = np.zeros((10240, 16), dtype=np.float64)
        matrices_wrist[i, j] = np.zeros((10240, 12), dtype=np.float64)

foldername = []
filename = []
flag = 0
count = 0

for isession in range(1, nsession + 1):  # Total number of sessions per participant
    converted_folder = f"Session{isession}_converted"
    os.makedirs(os.path.join("Output BM", converted_folder), exist_ok=True)

    for isub in range(1, nsub + 1):
        foldername = f"session{isession}_participant{isub}"

        for igesture in range(1, ngesture + 2):  # +1 to include rest gesture
            for itrial in range(1, ntrials + 1):
                filename = f"session{isession}_participant{isub}_gesture{igesture}_trial{itrial}"
                filepath = os.path.join(os.getcwd(), f"Session{isession}", foldername, filename)

                # Load WFDB data
                record = wfdb.rdrecord(filepath)

                # Extract signals and other information
                data_emg = record.p_signal
                fs = record.fs

                # Extract forearm and wrist data based on channel mappings
                data_forearm = data_emg[:, forearm_channels.astype(bool)]
                data_wrist = data_emg[:, wrist_channels.astype(bool)]

                # Assuming DATA_FOREARM and DATA_WRIST are lists
                matrices_forearm[itrial - 1, igesture - 1] = data_forearm
                matrices_wrist[itrial - 1, igesture - 1] = data_wrist

        count += 1
        print(f"Converted: {count} of {nsub * nsession} files")
        # Create a dictionary to hold the data

        savemat(os.path.join("Output BM", converted_folder, f"{foldername}.mat"),
                {"DATA_FOREARM": matrices_forearm, "DATA_WRIST": matrices_wrist})
