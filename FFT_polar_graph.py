import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import find_peaks
import statistics as s
import scipy.stats as stats

prevlx = 0
prevly = 0
prevrx = 0
prevry =0

l_iris_anglesx = []
l_iris_anglesy = []
l_iris_radius = []

r_iris_anglesx = []
r_iris_anglesy = []
r_iris_radius = []

l_angular_movement = []
r_angular_movement = []
i=0

delta_left_iris_x = []
delta_left_iris_y = []
delta_right_iris_x = []
delta_right_iris_y = []

delta_left_iris_x = []
delta_left_iris_y = []
delta_right_iris_x = []
delta_right_iris_y = []

nl_iris_anglesx = []
nl_iris_anglesy = []
nl_iris_radius = []

nr_iris_anglesx = []
nr_iris_anglesy = []
nr_iris_radius = []

file1 = open('test_left.csv', 'r') 
file2 = open('test_right.csv', 'r') 
# file3 = open('test_left.csv', 'r') 
# file4 = open('test_right.csv', 'r')
try:
        for line1 in file1:
            line_values1 = line1.split(',')
            l_anglex = float(line_values1[0]) #l_cx
            l_angley = float(line_values1[1]) #l_cy
            l_radius = float(line_values1[2]) #l_radiu
        
            l_iris_anglesx.append((l_anglex))
            l_iris_anglesy.append((l_angley))
            l_iris_radius.append((l_radius))

        for line2 in file2:
            line_values2 = line2.split(',')
            r_anglex = float(line_values2[0])
            r_angley = float(line_values2[1])
            r_radius = float(line_values2[2])

            r_iris_anglesx.append((r_anglex))
            r_iris_anglesy.append((r_angley))
            r_iris_radius.append((r_radius))

        # for line3 in file3:
        #     line_values3 = line3.split(',')
        #     nl_anglex = float(line_values3[0])
        #     nl_angley = float(line_values3[1])
        #     nl_radius = float(line_values3[2])

        #     nl_iris_anglesx.append((nl_anglex))
        #     nl_iris_anglesy.append((nl_angley))
        #     nl_iris_radius.append((nl_radius))

        # for line4 in file4:
        #     line_value4 = line4.split(',')
        #     nr_anglex = float(line_values3[0])
        #     nr_angley = float(line_values3[1])
        #     nr_radius = float(line_values3[2])

        #     nr_iris_anglesx.append((nr_anglex))
        #     nr_iris_anglesy.append((nr_angley))
        #     nr_iris_radius.append((nr_radius))

except FileExistsError as e:
        print(e)
except Exception as e:
        print(e)
else:
        file1.close()
        file2.close()
        # file3.close()
        # file4.close()
for a in range(0,len(l_iris_anglesx)):
        if prevlx == 0 and prevly == 0 and prevrx == 0 and prevry == 0:
            delta_left_iris_x.append(l_iris_anglesx[a])
            delta_left_iris_y.append(l_iris_anglesy[a])
            delta_right_iris_x.append(r_iris_anglesx[a])
            delta_right_iris_y.append(r_iris_anglesy[a])

            prevlx=l_iris_anglesx[a]
            prevly=l_iris_anglesy[a]
            prevrx=r_iris_anglesx[a]
            prevry=r_iris_anglesy[a]
        else:
            delta_left_iris_x.append(l_iris_anglesx[a] - prevlx)
            delta_left_iris_y.append(l_iris_anglesy[a] - prevly)
            delta_right_iris_x.append(r_iris_anglesx[a] - prevrx)
            delta_right_iris_y.append(r_iris_anglesy[a] - prevry)
                
            prevlx=l_iris_anglesx[a]
            prevly=l_iris_anglesy[a]
            prevrx=r_iris_anglesx[a]
            prevry=r_iris_anglesy[a]
        # print(len(delta_left_iris_x))
        # print(delta_left_iris_y)
        # print(delta_right_iris_x)
        # print(delta_right_iris_y)
#(Math.atan2(x, y) * 180) / Math.PI; return dregree
while i < len(l_iris_anglesx):
    templ = (math.atan2( delta_left_iris_y[i],delta_left_iris_x[i])) #return radians
    tempr = (math.atan2(delta_right_iris_y[i], delta_right_iris_x[i])) # used math.dregrees to return dregree
    l_angular_movement.append(templ)
    r_angular_movement.append(tempr)
    i+=1
frame_indices1 = range(len(l_angular_movement))
frame_indices2 = range(len(r_angular_movement))


count = 0
j = 1
freqfunc=[0]
ampl= []
ampr = []
peakleft = []
peakright = []

n = 512 
delta_t = 1/30 
delta_tXn = delta_t*n 

l_fft = fft(l_angular_movement)
r_fft = fft(r_angular_movement)

while count < n:
    temp1 = abs(l_fft[count])
    temp2 = abs(r_fft[count]) 

    ampl.append(temp1)
    ampr.append(temp2) 
    count+=1

while j < n:
    hp0 = freqfunc[j-1]+1/delta_tXn
    freqfunc.append(float(format(hp0,".6f")))
    j+=1 # 0 <----> +n

freqfunc = freqfunc[12:-12]
ampl = ampl[12:-12]
ampr = ampr[12:-12]

l_amppeaks, _ = find_peaks(ampl)
r_amppeaks, _ = find_peaks(ampr)
for a in l_amppeaks:
    peakleft.append(ampl[a])
for b in r_amppeaks:
    peakright.append(ampr[b])
l_mean = s.mean(peakleft)
r_mean = s.mean(peakright)

l_std = s.stdev(peakleft)
r_std = s.stdev(peakright)
sqrtSampleSize = (math.sqrt(delta_tXn))

templx = l_std/sqrtSampleSize
temprx = r_std/sqrtSampleSize

upb_l = l_mean + (1.645*(templx))
lwb_l = l_mean - (1.645*(templx))

upb_r = r_mean + (1.645*(temprx))
lwb_r = r_mean - (1.645*(temprx))

print("left eye mean: ", l_mean)
print("upper bound left eye: ", upb_l)
print("lower bound left eye: ", lwb_l)

print("right eye mean: ", r_mean)
print("upper bound right eye: ",upb_r)
print("lower bound right eye: ",lwb_r)

# FFT plot
plt.subplot(2, 1, 1)
plt.plot(freqfunc,ampl, color='black')
plt.plot(np.array(freqfunc)[l_amppeaks], np.array(ampl)[l_amppeaks], 'rx', label='Peaks in ampleft')
plt.axhline(y=l_mean, color='b',label = "mean")
plt.axhline(y=upb_l, color='red',label = "amp upper bound")
plt.axhline(y=lwb_l, color='green',label = "amp lower bound")

plt.title("FFT patient of Left Iris")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(freqfunc,ampr, color='black')
plt.plot(np.array(freqfunc)[r_amppeaks], np.array(ampr)[r_amppeaks], 'rx', label='Peaks in ampleft')
plt.axhline(y=r_mean, color='b',label = "mean")
plt.axhline(y=upb_r, color='red',label = "amp upper bound")
plt.axhline(y=lwb_r, color='green',label = "amp lower bound")

plt.title("FFT patient of Left Iris")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)


# # just plot normal no fft
# plt.plot(frame_indices1, l_angular_movement)
# plt.plot(frame_indices2, r_angular_movement)
# plt.title('Angular Movement of Left Iris on X-axis')
# plt.xlabel('Frame')
# plt.ylabel('Angular Movement (degrees)')
# plt.grid(True)
plt.show()