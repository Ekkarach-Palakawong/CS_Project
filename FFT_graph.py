import matplotlib.pyplot as plt
import numpy as np
import math as m
from scipy.fft import fft
from scipy.signal import find_peaks
from scipy.signal import argrelextrema

ampx = np.array([])
ampy = np.array([])
freqx = np.array([0])
freqy = np.array([0])

count = 1
con_lv = 90/100

left_angle = []
#iris_positions_left_y = []

right_angle = []
#iris_positions_right_y = []

file1 = open('test_right.csv', 'r')
file2 = open('test_left.csv', 'r')
file3 = open('totalframe.csv', 'r')
try:
    for center_right in file1:
        #line_values = center_right.split(',')
        #x_value = float(line_values[0])
        #y_value = float(line_values[1])
        left_angle.append(center_right)
        #iris_positions_right_y.append(y_value)
    for center_left in file2:
        #line_values = center_left.split(',')
        #x_value = float(line_values[0])
        #y_value = float(line_values[1])
        right_angle.append(center_left)
        #iris_positions_left_y.append(y_value)
    for line in file3:
        line_values = line.split(',')
        ttl_Frame = float(line_values[0])
        ttl_time = float(line_values[1])
        fps = float(line_values[2])

except FileExistsError as e:
    print(e)
except Exception as e:
    print(e)
else:
    file1.close()   
    file2.close()
    file3.close()

n = 512 #sample size
delta_t = 1/fps #sampling time
delta_tXn = delta_t*n # this is used to find a freq
iris_positions_left = np.array(left_angle)
#iris_positions_lefty = np.array(iris_positions_left_y)

iris_positions_right = np.array(right_angle)
#iris_positions_righty = np.array(iris_positions_right_y)

fft_left = fft(iris_positions_left)  # FFT for x-coordinates of left iris
#fft_left_y = fft(iris_positions_lefty)  # FFT for y-coordinates of left iris
fft_right = fft(iris_positions_right)  # FFT for x-coordinates of right iris
#fft_right_y = fft(iris_positions_righty)  # FFT for y-coordinates of right iris

for i in fft_left,:
    ampx = np.append(ampx, abs(i))
for j in fft_right:
    ampy = np.append(ampy, abs(j))
# t=0
# while t < n:
#     ampx = np.append(ampx, abs(fft_left_x[t]))
#     ampy = np.append(ampy, abs(fft_right_x[t]))
#     t+=1
#debug
# for test1,test2 in zip(ampx,ampy):
#     print(test1==test2) #all False

# Calculate the frequency domain
while count < n:
     freqx = np.append(freqx, freqx[count-1]+(1/delta_tXn))
     freqy = np.append(freqy, freqy[count-1]+(1/delta_tXn))
     count+=1 # 0 <----> +n

#freq = np.fft.fftfreq(n, 1 / delta_t) #-n <---> 0 <---> +n

left_E = np.concatenate((freqx, ampx))
right_E = np.concatenate((freqy, ampy))

# left_E3 = np.concatenate((freqx, ampx))
# right_E4 = np.concatenate((freqy, ampy))

c_max_index = argrelextrema(left_E, np.greater, order=1)

peaks_left, _ = find_peaks(left_E)
peaks_right, _ = find_peaks(right_E)

mean_pl = np.mean(peaks_left)
mean_pr = np.mean(peaks_right)
std_pl = np.std(peaks_left)
std_pr = np.std(peaks_right)

#print(con_lv/2) #look in Z table
temp = (m.sqrt(n))
eu_l = 1.64*(std_pl/temp) #Experimental Uncertainty
eu_r = 1.64*(std_pr/temp)
upb_l = mean_pl + eu_l #upper bound
lwb_l = mean_pl - eu_l
upb_r = mean_pr + eu_r #lower bound
lwb_r = mean_pr - eu_r
print("upper bound Left Eye: ", upb_l)
print("lower bound Left Eye: ", lwb_l)
print("upper bound Right Eye: ", upb_r)
print("lower bound Right Eye: ", lwb_r)
print("mean peak left eye: {}".format(np.mean(peaks_left)))
print("mean peak right eye: {}".format(np.mean(peaks_right)))
print("std peak left eye: {}".format(np.std(peaks_left)))
print("std peak right eye: {}".format(np.std(peaks_right)))

#debug
# y=0
# while y < 1058:
#     print(left_E1[y]==left_E3[y],y)
#     print(right_E2[y]==right_E4[y],y)
#     y+=1
#print(freqx == freq)

# for temp1,temp2 in zip(left_E,right_E):
#     print("%f____%f"%(temp1,temp2))


plt.figure(figsize=(10,6))

# plt.plot(left_E)
# plt.scatter(c_max_index[0],left_E[c_max_index[0]],linewidth=0.3, s=50, c='r')

plt.subplot(2, 1, 1)
plt.plot(left_E)
plt.plot(peaks_left, left_E[peaks_left], "x")
plt.title("FFT of Left Iris X-coordinate")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(right_E)
plt.plot(peaks_right, right_E[peaks_right], "x")
plt.title("FFT of Right Iris X-coordinate")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

# plt.subplot(2, 2, 3)
# plt.plot(freqrx, np.abs(amprx))
# plt.title("FFT of Left Iris Y-coordinate")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude")

# plt.subplot(2, 2, 4)
# plt.plot(freqry, np.abs(ampry))
# plt.title("FFT of Right Iris Y-coordinate")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()