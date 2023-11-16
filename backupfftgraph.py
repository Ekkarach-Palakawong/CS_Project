import matplotlib.pyplot as plt
import numpy as np
import math as m
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks

ampleft = np.array([])
ampright = np.array([])
freqfunc = np.array([0])


Nampleft = np.array([])
Nampright = np.array([])

i = 0
count = 1
Ncount = 1

left_angle = []
right_angle = []
normalleft = []
normalright = []

file1 = open('left1_Lefteye.csv', 'r')
file2 = open('left1_Righteye.csv', 'r')
file3 = open('normal2_Lefteye.csv', 'r')
file4 = open('normal2_Righteye.csv', 'r')

try:
    for index1 in file1:
        left_angle.append(index1)
    for index2 in file2:
        right_angle.append(index2)
    for index3 in file3:
        normalleft.append(index3)
    for index4 in file4:
        normalright.append(index4)

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

normalleft = np.array(normalleft)
normalright = np.array(normalright)

left_angle = np.array(left_angle)
right_angle = np.array(right_angle)

N_fftleft = fft(normalleft)
N_fftright = fft(normalright)

fft_left = fft(left_angle)
fft_right = fft(right_angle)

while i < n:
    ampleft = np.append(ampleft, abs(fft_left[i]))
    ampright = np.append(ampright, abs(fft_right[i]))
    Nampleft = np.append(Nampleft, abs(N_fftleft[i]))
    Nampright = np.append(Nampright, abs(N_fftright[i]))
    i+=1
# Calculate the frequency domain
while count < n:
    hp1 = freqfunc[count-1]+1/delta_tXn
    freqfunc = np.append(freqfunc, hp1)
    count+=1 # 0 <----> +n

#freq = np.fft.fftfreq(n, 1 / delta_t) #-n <---> 0 <---> +n


# ampleft = ampleft[6:-8]
# Nampleft = Nampleft[6:-8]
left_E = np.concatenate((freqfunc, ampleft))
right_E = np.concatenate((freqfunc, ampright))
# print(ampleft)
N_left_E = np.concatenate((freqfunc, Nampleft))
N_right_E = np.concatenate((freqfunc, Nampright))

#c_max_index = argrelextrema(left_E, np.greater, order=1)

peaks_left, _ = find_peaks(left_E)  # ' _ ' is mean ignore value if it return multiple value. "
peaks_right, _ = find_peaks(right_E)

Npeaks_left, _ = find_peaks(N_left_E)
Npeaks_right, _ = find_peaks(N_right_E)

#peak domain
mean_pl = np.mean(peaks_left)
mean_pr = np.mean(peaks_right)
std_pl = np.std(peaks_left)
std_pr = np.std(peaks_right)

N_mean_pl = np.mean(Npeaks_left)
N_mean_pr = np.mean(Npeaks_right)
N_std_pl = np.std(Npeaks_left)
N_std_pr = np.std(Npeaks_right)

#amplitude dommain
mean_al = np.mean(ampleft)
mean_ar = np.mean(ampright)
std_al = np.std(ampleft)
std_ar = np.std(ampright)

N_mean_al = np.mean(Nampleft)
N_mean_ar = np.mean(Nampright)
N_std_al = np.std(Nampleft)
N_std_ar = np.std(Nampright)

#print(con_lv/2) #look in Z table
sqrtSampleSize = (m.sqrt(n))

CI_al = stats.norm.interval(0.90, loc=mean_al, scale=std_al/sqrtSampleSize)
CI_ar = stats.norm.interval(0.90, loc=mean_ar, scale=std_ar/sqrtSampleSize)
lwb_al ,upb_al = CI_al
lwb_ar ,upb_ar = CI_ar

CI_pl = stats.norm.interval(0.90, loc=mean_pl, scale=std_pl/sqrtSampleSize)
CI_pr = stats.norm.interval(0.90, loc=mean_pr, scale=std_pr/sqrtSampleSize)
lwb_pl ,upb_pl = CI_pl
lwb_pr ,upb_pr = CI_pr

N_CI_al = stats.norm.interval(0.90, loc=N_mean_al, scale=N_std_al/sqrtSampleSize)
N_CI_ar = stats.norm.interval(0.90, loc=N_mean_ar, scale=N_std_ar/sqrtSampleSize)
N_lwb_al ,N_upb_al = N_CI_al
N_lwb_ar ,N_upb_ar = N_CI_ar

N_CI_pl = stats.norm.interval(0.90, loc=N_mean_pl, scale=N_std_pl/sqrtSampleSize)
N_CI_pr = stats.norm.interval(0.90, loc=N_mean_pr, scale=N_std_pr/sqrtSampleSize)
N_lwb_pl ,N_upb_pl = N_CI_pl
N_lwb_pr ,N_upb_pr = N_CI_pr

# #patient***********************************************************************
#left eye peaks
print("ค่าเฉลี่ยของจุด peak ตาซ้าย: {}".format(mean_pl))
print("ส่วนเบี่ยงเบนมาตรฐาน peak ตาซ้าย: {}".format(std_pl))
print("ช่วงความเชื่อมั่นของ peak ที่ระดับ 90% ของตาซ้าย: ",CI_pl)
print("\n")

# #right eye peaks
# print("ค่าเฉลี่ยของจุด peak ตาขวา: {}".format(mean_pr))
# print("ส่วนเบี่ยงเบนมาตรฐาน peak ตาขวา: {}".format(std_pr))
# print("ช่วงความเชื่อมั่นของ peak ที่ระดับ 90% ของตาขวา: ",CI_pr)
# print("\n")

# #left eye amplitude
print("ค่าเฉลี่ยของ amplitude ตาซ้าย: {}".format(mean_al))
print("ส่วนเบี่ยงเบนมาตรฐาน amplitude ตาซ้าย: {}".format(std_al))
print("ช่วงความเชื่อมั่นของ amplitude ที่ระดับ 90% ของตาซ้าย: ",CI_al)
print("\n")

#right eye amplitude
# print("ค่าเฉลี่ยของ amplitude ตาขวา: {}".format(mean_ar))
# print("ส่วนเบี่ยงเบนมาตรฐาน amplitude ตาขวา: {}".format(std_ar))
# print("ช่วงความเชื่อมั่นของ amplitude ที่ระดับ 90% ของตาขวา: ",CI_ar)

# #normalpeople***********************************************************************
#left eye peaks
print("ค่าเฉลี่ยของจุด peak ตาซ้าย: {}".format(N_mean_pl))
print("ส่วนเบี่ยงเบนมาตรฐาน peak ตาซ้าย: {}".format(N_std_pl))
print("ช่วงความเชื่อมั่นของ peak ที่ระดับ 90% ของตาซ้าย: ",N_CI_pl)
print("\n")

#right eye peaks
# print("ค่าเฉลี่ยของจุด peak ตาขวา: {}".format(N_mean_pr))
# print("ส่วนเบี่ยงเบนมาตรฐาน peak ตาขวา: {}".format(N_std_pr))
# print("ช่วงความเชื่อมั่นของ peak ที่ระดับ 90% ของตาขวา: ",N_CI_pr)
# print("\n")

# #left eye amplitude
print("ค่าเฉลี่ยของ amplitude ตาซ้าย: {}".format(N_mean_al))
print("ส่วนเบี่ยงเบนมาตรฐาน amplitude ตาซ้าย: {}".format(N_std_al))
print("ช่วงความเชื่อมั่นของ amplitude ที่ระดับ 90% ของตาซ้าย: ",N_CI_al)
print("\n")

#right eye amplitude
# print("ค่าเฉลี่ยของ amplitude ตาขวา: {}".format(N_mean_ar))
# print("ส่วนเบี่ยงเบนมาตรฐาน amplitude ตาขวา: {}".format(N_std_ar))
# print("ช่วงความเชื่อมั่นของ amplitude ที่ระดับ 90% ของตาขวา: ",N_CI_ar)

# plot graph ***********************************************************************
#plt.subplot(2, 1, 1)
plt.plot(freqfunc[10:],ampleft[10:], color = 'black')
plt.plot(freqfunc[10:],Nampleft[10:], color = 'darkgray')
#plt.plot(peaks_left, left_E[peaks_left], "x")
#patient
plt.axhline(y=upb_al, color='red',label = "patient amp upper bound")
plt.axhline(y=lwb_al, color='green',label = "patient amp lower bound")

plt.axvline(x=upb_pl, color='orange',label = "patient peak upper bound")
plt.axvline(x=lwb_pl, color='purple',label = "patient peak lower bound")
#normal
plt.axhline(y=N_upb_al, color='darkblue',label = "normal amp upper bound")
plt.axhline(y=N_lwb_al, color='darkturquoise',label = "normal amp lower bound")

plt.axvline(x=N_upb_pl, color='yellow',label = "normal peak upper bound")
plt.axvline(x=N_lwb_pl, color='deeppink',label = "normal peak lower bound")

plt.title("FFT of Left Iris")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(freqfunc[10:],ampleft[10:], color = 'black')
# plt.plot(freqfunc[10:],Nampleft[10:], color = 'darkgray')
# plt.plot(peaks_right, right_E[peaks_right], "x")
# patient
# plt.axhline(y=upb_ar, color='red',label = "patient amp upper bound")
# plt.axhline(y=lwb_ar, color='green',label = "patient amp lower bound")

# plt.axvline(x=upb_pr, color='orange',label = "patient peak upper bound")
# plt.axvline(x=lwb_pr, color='purple',label = "patient peak lower bound")
# normal
# plt.axhline(y=N_upb_ar, color='darkblue',label = "normal amp upper bound")
# plt.axhline(y=N_lwb_ar, color='darkturquoise',label = "normal amp lower bound")

# plt.axvline(x=N_upb_pr, color='yellow',label = "normal peak upper bound")
# plt.axvline(x=N_lwb_pr, color='deeppink',label = "normal peak lower bound")

# plt.title("FFT of Right Iris")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.legend()
plt.show()