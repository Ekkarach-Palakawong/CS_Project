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
amplefty = []
amprightx = []
amprighty = []
Nampleftx = []
Namplefty = []
Namprightx = []
Namprighty = []

peakleftx = []
peaklefty = []
peakrightx = []
peakrighty = []
Npeakleftx = []
Npeaklefty = []
Npeakrightx = []
Npeakrighty = []

i = 0
count = 1
Ncount = 1

left_anglex = []
left_angley = []
right_anglex = []
right_angley = []
normalleftx = []
normallefty = []
normalrightx = []
normalrighty = []

path = "C:/Users/pnaSu/Desktop/openCV_project/csv/"

file1 = open(path+'Nervouscenteriris_lefteye.csv', 'r')
file2 = open(path+'Nervouscenteriris_righteye.csv', 'r')
file3 = open(path+'left2centeriris_lefteye.csv', 'r')
file4 = open(path+'left2centeriris_righteye.csv', 'r')

try:
    for index1 in file1:
        index1 = index1.split(',')
        left_anglex.append(index1[0])
        left_angley.append(index1[1])
        #left_angle.append(format(float(index1),".6f"))
    for index2 in file2:
        index2 = index2.split(',')
        right_anglex.append(index2[0])
        right_angley.append(index2[1])
        #right_angle.append(format(float(index2),".6f"))
    for index3 in file3:
        index3 = index3.split(',')
        normalleftx.append(index3[0])
        normallefty.append(index3[1])
    for index4 in file4:
        index4 = index4.split(',')
        normalrightx.append(index4[0])
        normalrighty.append(index4[1])

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
fft_Nlefty = fft(normallefty)
fft_Nrightx = fft(normalrightx)
fft_Nrighty = fft(normalrighty)

fft_leftx = fft(left_anglex)
fft_lefty = fft(left_angley)
fft_rightx = fft(right_anglex)
fft_righty = fft(right_angley)

while i < n:
    hp1x = abs(fft_leftx[i])
    hp1y = abs(fft_lefty[i])

    hp2x = abs(fft_rightx[i])
    hp2y = abs(fft_righty[i])
        
    hp3x = abs(fft_Nleftx[i])
    hp3y = abs(fft_Nlefty[i])

    hp4x = abs(fft_Nrightx[i])
    hp4y = abs(fft_Nrighty[i])

    ampleftx.append(hp1x)
    amplefty.append(hp1y)

    amprightx.append(hp2x)
    amprighty.append(hp2y)

    Nampleftx.append(hp3x)
    Namplefty.append(hp3y)

    Namprightx.append(hp4x)
    Namprighty.append(hp4y)
    i+=1

# Calculate the frequency domain
while count < n:
    hp0 = freqfunc[count-1]+(1/delta_tXn)
    freqfunc.append(float(format(hp0,".6f")))
    # freqfunc = np.append(freqfunc, hp0)
    count+=1 # 0 <----> +n

freqfunc = freqfunc[:-12]

ampleftx = ampleftx[12:]
amplefty = amplefty[12:]
amprightx = amprightx[12:]
amprighty = amprighty[12:]

Nampleftx = Nampleftx[12:]
Namplefty = Namplefty[12:]
Namprightx = Namprightx[12:]
Namprighty = Namprighty[12:]
# print(len(freqfunc))
# print(len(ampleft))
# print(len(Nampleft))

#get firstfive freq. and amplitude
for p in freqfunc:
    if p <= 5:
        FF_freq.append(p)
    else:
        break
# ner_freq=[]
# temp1=[]
# temp2=[]
# for q in freqfunc: # 1-6
#     if q >=1 and q <= 6:
#         ner_freq.append(q)
#     elif q >=0 and q <= 1: #0-1
#         temp1.append(q)
#     elif q > 6: #6 above
#         temp2.append(q)
#     else:
#         break

# print(FF_freq)
# print(len(FF_freq))

#0-5
ampleftx= ampleftx[:len(FF_freq)]
amplefty= amplefty[:len(FF_freq)]
amprightx= amprightx[:len(FF_freq)]
amprighty= amprighty[:len(FF_freq)]

Nampleftx= Nampleftx[:len(FF_freq)]
Namplefty= Namplefty[:len(FF_freq)]
Namprightx= Namprightx[:len(FF_freq)]
Namprighty= Namprighty[:len(FF_freq)]
#1-6
# Nampleftx= Nampleftx[len(temp1):-len(temp2)]
# Namplefty= Namplefty[len(temp1):-len(temp2)]
# Namprightx= Namprightx[len(temp1):-len(temp2)]
# Namprighty= Namprighty[len(temp1):-len(temp2)]

# Find peaks in the amplitude
# ampeaks_leftx, _ = find_peaks(ampleftx)
# ampeaks_lefty, _ = find_peaks(amplefty)  # Find peaks in ampleft array

# ampeaks_rightx, _ = find_peaks(amprightx)
# ampeaks_righty, _ = find_peaks(amprighty)

# normal_ampeaks_leftx, _ = find_peaks(Nampleftx)  # Find peaks in Nampleft array
# normal_ampeaks_lefty, _ = find_peaks(Namplefty)

# normal_ampeaks_rightx, _ = find_peaks(Namprightx)  
# normal_ampeaks_righty, _ = find_peaks(Namprighty)

# for a in ampeaks_leftx:
#     peakleftx.append( ampleftx [a] )
# for b in ampeaks_lefty:
#     peaklefty.append( amplefty [b] )
# for c in ampeaks_rightx:
#     peakrightx.append( amprightx [c] )
# for d in ampeaks_righty:
#     peakrighty.append( amprighty [d] )

# for e in normal_ampeaks_leftx:
#     Npeakleftx.append( Nampleftx [e] )
# for f in normal_ampeaks_lefty:
#     Npeaklefty.append( Namplefty [f] )
# for g in normal_ampeaks_rightx:
#     Npeakrightx.append( Namprightx [g] )
# for h in normal_ampeaks_righty:
#     Npeakrighty.append( Namprighty [h] )


#firstfive_element
# FF_peakleftx = peakleftx [ :len(FF_freq) ]
# FF_peaklefty = peaklefty [ :len(FF_freq) ]
# FF_peakrightx = peakrightx [ :len(FF_freq) ]
# FF_peakrighty = peakrighty [ :len(FF_freq) ]

# FF_normal_peakleftx = normal_peakleftx [ :len(FF_freq) ]
# FF_normal_peaklefty = normal_peaklefty [ :len(FF_freq) ]
# FF_normal_peakrightx = normal_peakrightx [ :len(FF_freq) ]
# FF_normal_peakrighty = normal_peakrighty [ :len(FF_freq) ]

mean_peakleftx = s.mean(ampleftx)
mean_peaklefty = s.mean(amplefty)
mean_peakrightx = s.mean(amprightx)
mean_peakrighty = s.mean(amprighty)

mean_Npeakleftx = s.mean(Nampleftx)
mean_Npeaklefty = s.mean(Namplefty)
mean_Npeakrightx = s.mean(Namprightx)
mean_Npeakrighty = s.mean(Namprighty)

std_peakleftx = s.stdev(ampleftx)
std_peaklefty = s.stdev(amplefty)
std_peakrightx = s.stdev(amprightx)
std_peakrighty = s.stdev(amprighty)

std_Npeakleftx = s.stdev(Nampleftx)
std_Npeaklefty = s.stdev(Namplefty)
std_Npeakrightx = s.stdev(Namprightx)
std_Npeakrighty = s.stdev(Namprighty)

# total = 0
# variance = 0
# for y in peakleftx:
#     total=total+y
# manul_mean = total/len(peakleftx)
# for b in peakleftx:
#     variance = variance+(b-manul_mean)**2
# manual_variance=variance/(len(peakleftx)-1)
# manual_std = m.sqrt(manual_variance)
# print("manul_mean: ",manul_mean)
# print("manual_variance: ",manual_variance)
# print("SD: ",manual_std)

# upper_bound = manul_mean+manual_variance
# lower_bound = manul_mean- manual_variance

# print("manual_upper_bound: ",upper_bound )
# print("manual_lower_bound: ",lower_bound)
# print(np.array(freqfunc)[ampeaks])
# print(np.array(ampleft)[ampeaks])
# print(manul_mean)
# print(manual_std)
# print(manual_variance)

# mean_al = s.mean(peakleftx)
# std_al = s.stdev(peakleftx)
# print("mean_al: ", mean_al)
# print("std_al: ",std_al)
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
# confiden_lv = 0.90
# sample_std_error = manual_std / m.sqrt(delta_tXn)
# z_critical = stats.norm.ppf((1 + confiden_lv) / 2)

# margin_of_error = z_critical * sample_std_error
# lower_bound = mean_al - margin_of_error
# upper_bound = mean_al + margin_of_error
# print("z_critical: ", z_critical)
# print("margin_of_error: ", margin_of_error)
# print("upper_bound: ", upper_bound)
# print("lower_bound: ", lower_bound)

sqrtSampleSize = (m.sqrt(delta_tXn))

temp1x = std_peakleftx/sqrtSampleSize
temp1y = std_peaklefty/sqrtSampleSize
temp2x = std_peakrightx/sqrtSampleSize
temp2y = std_peakrighty/sqrtSampleSize

temp3x = std_Npeakleftx/sqrtSampleSize
temp3y = std_Npeaklefty/sqrtSampleSize
temp4x = std_Npeakrightx/sqrtSampleSize
temp4y = std_Npeakrighty/sqrtSampleSize
# temp4 = N_std_al/sqrtSampleSize

CI_lx = stats.norm.interval(0.975, loc = mean_peakleftx, scale = temp1x)
lwb_lx ,upb_lx = CI_lx
CI_ly = stats.norm.interval(0.975, loc = mean_peaklefty, scale = temp1y)
lwb_ly ,upb_ly = CI_ly

CI_rx = stats.norm.interval(0.975, loc = mean_peakrightx, scale = temp2x)
lwb_rx ,upb_rx = CI_rx
CI_ry = stats.norm.interval(0.975, loc = mean_peakrighty, scale = temp2y)
lwb_ry ,upb_ry = CI_ry

N_CI_lx = stats.norm.interval(0.975, loc = mean_Npeakleftx, scale = temp3x)
N_lwb_lx ,N_upb_lx = N_CI_lx
N_CI_ly = stats.norm.interval(0.975, loc = mean_Npeaklefty, scale = temp3y)
N_lwb_ly ,N_upb_ly = N_CI_ly

N_CI_rx = stats.norm.interval(0.975, loc = mean_Npeakrightx, scale = temp4x)
N_lwb_rx ,N_upb_rx = N_CI_rx
N_CI_ry = stats.norm.interval(0.975, loc = mean_Npeakrighty, scale = temp4y)
N_lwb_ry ,N_upb_ry = N_CI_ry

print('upper bound patient: ',upb_lx)
print("mean patient: ",mean_peakleftx)
print('lower bound patient: ',lwb_lx)
print("***********************")
print('upper bound normal: ',N_upb_lx)
print("mean normal people: ",mean_Npeakleftx)
print('lower bound normal: ',N_lwb_lx)


plt.subplot(2, 1, 1)
plt.ylim(0, 300)
plt.scatter(FF_freq,ampleftx, color='black')
plt.plot(FF_freq,ampleftx, color='gray')
#plt.plot(np.array(freqfunc)[ampeaks_leftx], np.array(ampleftx)[ampeaks_leftx], 'rx', label='Peaks in ampleft')
plt.axhline(y=upb_rx, color='red',label = "upper bound")
plt.axhline(y=mean_peakleftx, color='b',label = "mean")
plt.axhline(y=lwb_lx, color='green',label = "lower bound")

# plt.plot(ampleftx,freqfunc, color='black')
# plt.axvline(x=upb_lx, color='red',label = "amp upper bound")
# plt.axvline(x=mean_peakleftx, color='b',label = "amp upper bound")
# plt.axvline(x=lwb_lx, color='green',label = "amp upper bound")

plt.title("FFT patient of Right Iris")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# #normal
plt.subplot(2, 1, 2)
plt.ylim(0, 300)
plt.scatter(FF_freq,Nampleftx, color='sienna')
plt.plot(FF_freq,Nampleftx, color='black')
#plt.plot(freqfunc[normal_ampeaks_leftx], Nampleftx[normal_ampeaks_leftx], 'bx', label='Peaks in ampleft')
plt.axhline(y=N_upb_lx, color='purple',label = "upper bound")
plt.axhline(y=mean_Npeakleftx, color='b',label = "mean")
plt.axhline(y=N_lwb_lx, color='darkturquoise',label = "lower bound")

# plt.plot(Nampleftx,freqfunc, color='sienna')
# plt.axvline(x=N_upb_lx, color='purple',label = "normal amp upper bound")
# plt.axvline(x=mean_Npeakleftx, color='b',label = "mean")
# plt.axvline(x=N_lwb_lx, color='darkturquoise',label = "normal amp lower bound")

plt.title("FFT nervous of Right Iris")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()