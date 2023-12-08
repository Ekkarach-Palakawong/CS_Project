import matplotlib.pyplot as plt
import math as m
#import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
import scipy.stats as stats
import statistics as s

FF_freq = []
freqfunc = [0]

ampleftx = []
amprightx = []

Nampleftx = []
Namprightx = []

peakleftx = []
peakrightx = []

Npeakleftx = []
Npeakrightx = []

i = 0
count = 1
Ncount = 1

left_anglex = []
right_anglex = []

normalleftx = []
normalrightx = []

path = "C:/Users/pnaSu/Desktop/openCV_project/csv/"

file1 = open(path+'left2centeriris_lefteye.csv', 'r')
file2 = open(path+'left2centeriris_righteye.csv', 'r')
file3 = open(path+'Nervouscenteriris_lefteye.csv', 'r')
file4 = open(path+'Nervouscenteriris_righteye.csv', 'r')

try:
    for index1 in file1:
        index1 = index1.split(',')
        left_anglex.append(index1[0])
        #left_angle.append(format(float(index1),".6f"))
    for index2 in file2:
        index2 = index2.split(',')
        right_anglex.append(index2[0])
        #right_angle.append(format(float(index2),".6f"))
    for index3 in file3:
        index3 = index3.split(',')
        normalleftx.append(index3[0])
    for index4 in file4:
        index4 = index4.split(',')
        normalrightx.append(index4[0])

except FileExistsError as e:
    print(e)
except Exception as e:
    print(e)
else:
    file1.close()
    file2.close()
    file3.close()
    file4.close()

n = 512 #sample size 512 256 128
delta_t = 1/30 #sampling time
delta_tXn = delta_t*n # this is used to find a freq

fft_Nleftx = fft(normalleftx)
fft_Nrightx = fft(normalrightx)

fft_leftx = fft(left_anglex)
fft_rightx = fft(right_anglex)

while i < n:
    hp1x = abs(fft_leftx[i])

    hp2x = abs(fft_rightx[i]) 
        
    hp3x = abs(fft_Nleftx[i])

    hp4x = abs(fft_Nrightx[i])

    ampleftx.append(hp1x)

    amprightx.append(hp2x)

    Nampleftx.append(hp3x)

    Namprightx.append(hp4x)

    i+=1

# Calculate the frequency domain
while count < n:
    hp0 = freqfunc[count-1]+(1/delta_tXn)
    freqfunc.append(float(format(hp0,".6f")))
    # freqfunc = np.append(freqfunc, hp0)
    count+=1 # 0 <----> +n

freqfunc = freqfunc[:-12]

ampleftx = ampleftx[12:]
amprightx = amprightx[12:]

Nampleftx = Nampleftx[12:]
Namprightx = Namprightx[12:]
# print(len(freqfunc))
# print(len(ampleft))
# print(len(Nampleft))

#get firstfive freq. and amplitude
for p in freqfunc:
    if p <= 5:
        FF_freq.append(p)
    else:
        break
# print(len(FF_freq))
ampleftx= ampleftx[:len(FF_freq)]
amprightx= amprightx[:len(FF_freq)]
Nampleftx= Nampleftx[:len(FF_freq)]
Namprightx= Namprightx[:len(FF_freq)]

# Find peaks in the amplitude
# ampeaks_leftx, _ = find_peaks(ampleftx)

# ampeaks_rightx, _ = find_peaks(amprightx)

# normal_ampeaks_leftx, _ = find_peaks(Nampleftx)  # Find peaks in Nampleft array

# normal_ampeaks_rightx, _ = find_peaks(Namprightx)  

# print(len(ampleft))
# print(len(Nampleft))
# print(len(ampeaks))
# print(len(normal_ampeaks))

# for a in ampleftx:
#     peakleftx.append( ampleftx [a] )

# for c in amprightx:
#     peakrightx.append( amprightx [c] )

# for e in Nampleftx:
#     Npeakleftx.append( Nampleftx [e] )

# for g in Namprightx:
#     Npeakrightx.append( Namprightx [g] )

mean_peakleftx = s.mean(ampleftx)
mean_peakrightx = s.mean(amprightx)

mean_Npeakleftx = s.mean(Nampleftx)
mean_Npeakrightx = s.mean(Namprightx)


std_peakleftx = s.stdev(ampleftx)
std_peakrightx = s.stdev(amprightx)

std_Npeakleftx = s.stdev(Nampleftx)
std_Npeakrightx = s.stdev(Namprightx)

sqrtSampleSize = (m.sqrt(delta_tXn))

temp1x = std_peakleftx/sqrtSampleSize
temp2x = std_peakrightx/sqrtSampleSize

temp3x = std_Npeakleftx/sqrtSampleSize
temp4x = std_Npeakrightx/sqrtSampleSize
# temp4 = N_std_al/sqrtSampleSize

CI_lx = stats.norm.interval(0.975, loc = mean_peakleftx, scale = temp1x)
lwb_lx ,upb_lx = CI_lx

CI_rx = stats.norm.interval(0.975, loc = mean_peakrightx, scale = temp2x)
lwb_rx ,upb_rx = CI_rx

N_CI_lx = stats.norm.interval(0.975, loc = mean_Npeakleftx, scale = temp3x)
N_lwb_lx ,N_upb_lx = N_CI_lx

N_CI_rx = stats.norm.interval(0.975, loc = mean_Npeakrightx, scale = temp4x)
N_lwb_rx ,N_upb_rx = N_CI_rx

print("mean patient: ",mean_peakleftx)
print('upper bound patient: ',upb_lx)
print('lower bound patient: ',lwb_lx)
print("***********************")
print("mean normal people: ",mean_Npeakleftx)
print('upper bound normal: ',N_upb_lx)
print('lower bound normal: ',N_lwb_lx)

plt.subplot(2, 1, 1)
plt.ylim(0, 2.6)
plt.scatter(FF_freq,ampleftx, color='black')
plt.plot(FF_freq,ampleftx, color='gray')
#plt.plot(np.array(freqfunc)[ampeaks_leftx], np.array(ampleftx)[ampeaks_leftx], 'rx', label='Peaks in ampleft')
plt.axhline(y=upb_lx, color='red',label = "amp upper bound")
plt.axhline(y=mean_peakleftx, color='b',label = "mean")
plt.axhline(y=lwb_lx, color='green',label = "amp lower bound")

# plt.plot(ampleftx,freqfunc, color='black')
# plt.axvline(x=upb_lx, color='red',label = "amp upper bound")
# plt.axvline(x=mean_peakleftx, color='b',label = "amp upper bound")
# plt.axvline(x=lwb_lx, color='green',label = "amp upper bound")

plt.title("FFT patient of Left Iris")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude")
plt.grid(True)

#normal
plt.subplot(2, 1, 2)
plt.ylim(0, 2.6)
plt.scatter(FF_freq,Nampleftx, color='sienna')
plt.plot(FF_freq,Nampleftx, color='black')
#plt.plot(freqfunc[normal_ampeaks_leftx], Nampleftx[normal_ampeaks_leftx], 'bx', label='Peaks in ampleft')
plt.axhline(y=N_upb_lx, color='purple',label = "normal amp upper bound")
plt.axhline(y=mean_Npeakleftx, color='b',label = "mean")
plt.axhline(y=N_lwb_lx, color='darkturquoise',label = "normal amp lower bound")

# plt.plot(Nampleftx,freqfunc, color='sienna')
# plt.axvline(x=N_upb_lx, color='purple',label = "normal amp upper bound")
# plt.axvline(x=mean_Npeakleftx, color='b',label = "mean")
# plt.axvline(x=N_lwb_lx, color='darkturquoise',label = "normal amp lower bound")

plt.title("FFT normal of Left Iris")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()