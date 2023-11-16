import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Sample x and y coordinate data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1.2, 3.4, 2.5, 4.7, 1.1, 2.9, 5.5, 4.2, 3.6, 5.9])

# Find peaks in y-values based on the x-values
peaks, _ = find_peaks(y, height=1, distance=1)

# Access the x and y coordinates of the detected peaks
peak_x = x[peaks]
peak_y = y[peaks]
print(_)
print(y[3])
print(y[6])
# Plot the x, y data and the detected peaks
plt.plot(x, y)
plt.scatter(peak_x, peak_y, c='red', marker='x', label='Peaks')
plt.legend()
plt.show()

# Print the x and y coordinates of the detected peaks
print("Peak x-coordinates:", peak_x)
print("Peak y-coordinates:", peak_y)
