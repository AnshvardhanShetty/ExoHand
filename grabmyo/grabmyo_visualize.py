import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

# Load the .mat file
mat_data = loadmat('Output BM\Session1_converted\session1_participant2.mat')

# Access the data
data = mat_data["DATA_FOREARM"]
print("Num Trials:", data.shape[0], "Num Gestures:", data.shape[1])

# Choose data from one run
data_to_plot = data[1, 3]
print("Timesteps:", data_to_plot.shape[0], "Num Channels:", data_to_plot.shape[1])

# randomly choose one channel
ichannel = np.random.randint(0, data_to_plot.shape[1])
time = range(data_to_plot.shape[0])
plt.figure(figsize=(10, 6))
plt.plot(time, data_to_plot[:, ichannel])
plt.title('Channel {}'.format(ichannel + 1))
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
