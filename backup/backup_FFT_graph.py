import matplotlib.pyplot as plt
import math as m
import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
import scipy.stats as stats
import statistics as s

ampleft = []
ampright = []
freqfunc = [0]
peakleft = []
normal_peakleft = []

Nampleft = []
Nampright = []

i = 0
count = 1
Ncount = 1

left_angle = []
right_angle = []
normalleft = []
normalright = []

file1 = open('left2_Lefteye.csv', 'r')
file2 = open('left2_Righteye.csv', 'r')
file3 = open('normal1_Lefteye.csv', 'r')
file4 = open('normal1_Righteye.csv', 'r')

try:
    for index1 in file1:
        index1 = index1.split(',')
        left_angle.append(format(float(index1[0]),".6f"))
        #left_angle.append(format(float(index1),".6f"))
    for index2 in file2:
        index2 = index2.split(',')
        right_angle.append(format(float(index2[0]),".6f"))
        #right_angle.append(format(float(index2),".6f"))
    for index3 in file3:
        normalleft.append(format(float(index3),".6f"))
    for index4 in file4:
        normalright.append(format(float(index4),".6f"))

except FileExistsError as e:
    print(e)
except Exception as e:
    print(e)
else:
    file1.close()
    file2.close()
    file3.close()
    file4.close()

n = 512 #sample size 256 128
delta_t = 1/30 #sampling time
delta_tXn = delta_t*n # this is used to find a freq
# normalleft = np.array(normalleft)
# normalright = np.array(normalright)
# left_angle = np.array(left_angle)
# right_angle = np.array(right_angle)

N_fftleft = fft(normalleft)
N_fftright = fft(normalright)

fft_left = fft(left_angle)
fft_right = fft(right_angle)

while i < n:
    hp1 = abs(fft_left[i])
    hp2 = abs(fft_right[i])
    hp3 = abs(N_fftleft[i])
    hp4 = abs(N_fftright[i])
    ampleft.append(float(format(hp1,".6f")))
    ampright.append(float(format(hp2,".6f")))
    Nampleft.append(float(format(hp3,".6f")))
    Nampright.append(float(format(hp4,".6f")))
    # ampleft = np.append(ampleft, hp1)
    # ampright = np.append(ampright, hp2)
    # Nampleft = np.append(Nampleft, hp3)
    # Nampright = np.append(Nampright, hp4)
    i+=1
# Calculate the frequency domain
while count < n:
    hp0 = freqfunc[count-1]+1/delta_tXn
    freqfunc.append(float(format(hp0,".6f")))
    # freqfunc = np.append(freqfunc, hp0)
    count+=1 # 0 <----> +n

freqfunc = freqfunc[12:-12]
ampleft = ampleft[12:-12]
#ampright = ampright[10:-10]
Nampleft = Nampleft[12: -12]
#Nampright = Nampright[10: -10]
# print(len(freqfunc))
# print(len(ampleft))
# print(len(Nampleft))

# Find peaks in the amplitude
ampeaks, properties1 = find_peaks(ampleft)  # Find peaks in ampleft array
# normal_ampeaks, properties2 = find_peaks(Nampleft)  # Find peaks in Nampleft array

# print(len(ampleft))
# print(len(Nampleft))
# print(len(ampeaks)) #171
# print(len(normal_ampeaks)) #160
for l in ampeaks:
    peakleft.append(ampleft[l])
# for a in normal_ampeaks:
#     normal_peakleft.append(Nampleft[a])
# peakleft  =np.append(peakleft, g1)
# normal_peakleft  =np.append(normal_peakleft, g2)
total = 0
variance = 0
for y in peakleft:
    total=total+y
manul_mean = total/len(peakleft)
for b in peakleft:
    variance = variance+(b-manul_mean)**2
manual_variance=variance/(len(peakleft)-1)
manual_std = m.sqrt(manual_variance)
print("manul_mean: ",manul_mean)
print("manual_variance: ",manual_variance)
print("SD: ",manual_std)
# upper_bound = manul_mean+manual_variance
# lower_bound = manul_mean- manual_variance
# print("manual_upper_bound: ",upper_bound )
# print("manual_lower_bound: ",lower_bound)
# print(np.array(freqfunc)[ampeaks])
# print(np.array(ampleft)[ampeaks])
# print(manul_mean)
# print(manual_std)
# print(manual_variance)
#amplitude dommain
mean_al = s.mean(peakleft)
std_al = s.stdev(peakleft)
print("mean_al: ", mean_al)
print("std_al: ",std_al)
# N_mean_al = s.mean(normal_peakleft)
# N_std_al = s.stdev(normal_peakleft)
# print(("N_mean_al: ", N_mean_al))
# print(("N_std_al: ",N_std_al))


# upper_bound = mean_al + (1.645*(manual_std/m.sqrt(n)))
# lower_bound = mean_al - (1.645*(manual_std/m.sqrt(n)))
# normal_upper_bound = N_mean_al + (1.645*(std_al/m.sqrt(delta_tXn)))
# normal_lower_bound = N_mean_al - (1.645*(std_al/m.sqrt(delta_tXn)))

# print("patien_upper_bound: ", upper_bound)
# print("patien_lower_bound: ", lower_bound)


# print("variance: ", upper_bound-mean_al)
confiden_lv = 0.90
sample_std_error = manual_std / m.sqrt(delta_tXn)
z_critical = stats.norm.ppf((1 + confiden_lv) / 2)

margin_of_error = z_critical * sample_std_error
lower_bound = mean_al - margin_of_error
upper_bound = mean_al + margin_of_error
print("z_critical: ", z_critical)
print("margin_of_error: ", margin_of_error)
print("upper_bound: ", upper_bound)
print("lower_bound: ", lower_bound)

# sqrtSampleSize = (m.sqrt(delta_tXn))

# temp1 = std_al/sqrtSampleSize
# # temp4 = N_std_al/sqrtSampleSize

# CI_al = stats.norm.interval(0.90, loc = mean_al, scale = temp1)
# lwb_al ,upb_al = CI_al

# N_CI_al = stats.norm.interval(0.99, loc = N_mean_al, scale = temp4)
# N_lwb_al ,N_upb_al = N_CI_al

# print("lower_bound: ",lwb_al, " ****** ","upper_bound: ", upb_al)

# Plot the amplitude and highlight peaks
# patient
print(len(freqfunc))
print(len(ampleft))
# plt.subplot(2, 1, 1)
plt.plot(freqfunc,ampleft, color='black')
plt.plot(np.array(freqfunc)[ampeaks], np.array(ampleft)[ampeaks], 'rx', label='Peaks in ampleft')
plt.axhline(y=mean_al, color='b',label = "mean")
plt.axhline(y=upper_bound, color='red',label = "amp upper bound")
plt.axhline(y=lower_bound, color='green',label = "amp lower bound")

plt.title("FFT patient of Left Iris")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)

#normal
# plt.subplot(2, 1, 2)
# plt.plot(freqfunc,Nampleft, color='darkgray')
# # plt.plot(, 'bx', label='Peaks in Nampleft')

# # plt.axhline(y=N_upb_al, color='darkblue',label = "normal amp upper bound")
# # plt.axhline(y=N_lwb_al, color='darkturquoise',label = "normal amp lower bound")

# plt.title("FFT normal of Left Iris")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude")
# plt.grid(True)
plt.show()