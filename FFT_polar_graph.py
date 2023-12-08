import math
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import find_peaks
import statistics as s
import scipy.stats as stats

i=0
n = 512 #sample size 512 256 128
delta_t = 1/30 #sampling time
delta_tXn = delta_t*n # this is used to find a freq

prevlx = 0
prevly = 0
prevrx = 0
prevry = 0 #patient
Nprevlx = 0 #normal or nerve
Nprevly = 0
Nprevrx = 0
Nprevry = 0

left_anglex = []
left_angley = []
#left_radius = [] #patient
nleft_anglex = [] #normal or nerve
nleft_angley = []
#nleft_radius = []

right_anglex = []
right_angley = []
#right_radius = [] #patient
nright_anglex = [] #normal or nerve
nright_angley = []
#nright_radius = []

delta_left_x = []
delta_left_y = []
delta_right_x = []
delta_right_y = [] #patient
Ndelta_left_x = [] #normal or nerve
Ndelta_left_y = []
Ndelta_right_x = []
Ndelta_right_y = []

left_angular_movement = []
right_angular_movement = [] #patient
Nleft_angular_movement = [] #normal or nerve
Nright_angular_movement = []

path = "C:/Users/pnaSu/Desktop/openCV_project/csv/"

file1 = open(path+'leftcenteriris_lefteye.csv', 'r')
file2 = open(path+'leftcenteriris_righteye.csv', 'r')
file3 = open(path+'normalcenteriris_lefteye.csv', 'r')
file4 = open(path+'normalcenteriris_righteye.csv', 'r')

try:
        for line1 in file1:
            line_values1 = line1.split(',')
            l_anglex = float(line_values1[0]) #l_cx
            l_angley = float(line_values1[1]) #l_cy
            #l_radius = float(line_values1[2]) #l_radiu
        
            left_anglex.append((l_anglex))
            left_angley.append((l_angley))
            #left_radius.append((l_radius))

        for line2 in file2:
            line_values2 = line2.split(',')
            r_anglex = float(line_values2[0])
            r_angley = float(line_values2[1])
            #r_radius = float(line_values2[2])

            right_anglex.append((r_anglex))
            right_angley.append((r_angley))
            #right_radius.append((r_radius))

        for line3 in file3:
            line_values3 = line3.split(',')
            nl_anglex = float(line_values3[0])
            nl_angley = float(line_values3[1])
            #nl_radius = float(line_values3[2])

            nleft_anglex.append((nl_anglex))
            nleft_angley.append((nl_angley))
            #nleft_radius.append((nl_radius))

        for line4 in file4:
            line_value4 = line4.split(',')
            nr_anglex = float(line_value4[0])
            nr_angley = float(line_value4[1])
            #nr_radius = float(line_values3[2])

            nright_anglex.append((nr_anglex))
            nright_angley.append((nr_angley))
            #nright_radius.append((nr_radius))

except FileExistsError as e:
        print(e)
except Exception as e:
        print(e)
else:
        file1.close()
        file2.close()
        file3.close()
        file4.close()

# print(len(left_anglex)) #546
# print(len(right_anglex)) #546
# print(len(nleft_anglex)) #542
# print(len(nright_anglex)) #542

# print(len(left_angley)) #546
# print(len(right_angley)) #546
# print(len(nleft_angley)) #542
# print(len(nright_angley)) #542

for a in range(0,520):
        if prevlx == 0 and prevly == 0 and prevrx == 0 and prevry == 0:
            # delta_left_x.append(left_anglex[a])
            # delta_left_y.append(left_angley[a])
            # delta_right_x.append(right_anglex[a])
            # delta_right_y.append(right_angley[a])

            # Ndelta_left_x.append(nleft_anglex[a])
            # Ndelta_left_y.append(nleft_angley[a])
            # Ndelta_right_x.append(nright_anglex[a])
            # Ndelta_right_y.append(nright_angley[a])

            prevlx=left_anglex[a]
            prevly=left_angley[a]
            prevrx=right_anglex[a]
            prevry=right_angley[a]

            Nprevlx=nleft_anglex[a]
            Nprevly=nleft_angley[a]
            Nprevrx=nright_anglex[a]
            Nprevry=nright_angley[a]
        else:
            delta_left_x.append(left_anglex[a] - prevlx)
            delta_left_y.append(left_angley[a] - prevly)
            delta_right_x.append(right_anglex[a] - prevrx)
            delta_right_y.append(right_angley[a] - prevry)

            Ndelta_left_x.append(nleft_anglex[a] - Nprevlx)
            Ndelta_left_y.append(nleft_angley[a] - Nprevly)
            Ndelta_right_x.append(nright_anglex[a] - Nprevrx)
            Ndelta_right_y.append(nright_angley[a] - Nprevry)
                
            prevlx=left_anglex[a]
            prevly=left_angley[a]
            prevrx=right_anglex[a]
            prevry=right_angley[a]

            Nprevlx=nleft_anglex[a]
            Nprevly=nleft_angley[a]
            Nprevrx=nright_anglex[a]
            Nprevry=nright_angley[a]

# print(len(delta_left_x)) #546
# print(len(delta_right_x)) #546
# print(len(Ndelta_left_x)) #546
# print(len(Ndelta_right_x)) #546

#(Math.atan2(x, y) * 180) / Math.PI; return dregree
while i < n: #arctan(delta y / delta x)
    templ = (math.atan2( delta_left_y[i], delta_left_x[i])) #return radians
    tempr = (math.atan2( delta_right_y[i], delta_right_x[i])) # used math.degrees to return dregree ( tan-1( delta y / deltar x))

    tempNl = (math.atan2( Ndelta_left_y[i], Ndelta_left_x[i]))
    tempNr = (math.atan2( Ndelta_right_y[i], Ndelta_right_x[i]))

    left_angular_movement.append(templ)
    right_angular_movement.append(tempr)

    Nleft_angular_movement.append(tempNl)
    Nright_angular_movement.append(tempNr)

    i+=1

# print(len(left_angular_movement)) #512
# print(len(right_angular_movement)) #512
# print(len(Nleft_angular_movement)) #512
# print(len(Nright_angular_movement)) #512

# leftframe_indices = range(len(left_angular_movement))
# rightframe_indices = range(len(right_angular_movement))
# Nleftframe_indices = range(len(Nleft_angular_movement))
# Nrightframe_indices = range(len(Nright_angular_movement))

FF_freq = []
freqfunc = [0]
ner_freq=[]

count = 0
j = 1

ampl= []
ampr = []
peakleft = []
peakright = []

Nampl= []
Nampr = []
Npeakleft = []
Npeakright = []

l_fft = fft(left_angular_movement)
r_fft = fft(right_angular_movement)

Nl_fft = fft(Nleft_angular_movement)
Nr_fft = fft(Nright_angular_movement)

# print(len(l_fft)) #512
# print(len(r_fft)) #512
# print(len(Nl_fft)) #512
# print(len(Nr_fft)) #512

while count < n:
    temp1 = abs(l_fft[count])
    temp2 = abs(r_fft[count])

    Ntemp1 = abs(Nl_fft[count])
    Ntemp2 = abs(Nr_fft[count]) 

    ampl.append(temp1)
    ampr.append(temp2) 

    Nampl.append(Ntemp1)
    Nampr.append(Ntemp2) 
    count+=1

# print(len(ampl)) #512
# print(len(ampr)) #512
# print(len(Nampl)) #512
# print(len(Nampr)) #512

while j < n:
    hp0 = freqfunc[j-1]+1/delta_tXn
    freqfunc.append(float(format(hp0,".6f")))
    j+=1 # 0 <----> +n

freqfunc = freqfunc[:-12]
ampl=ampl[12:]
ampr=ampr[12:]
Nampl=Nampl[12:]
Nampr=Nampr[12:]

# print(len(freqfunc)) #500
# print(len(ampl)) #500
# print(len(ampr)) #500
# print(len(Nampl)) #500
# print(len(Nampr)) #500

#get firstfive freq. and amplitude
# for p in freqfunc: #0-5
#     if p <= 5:
#         FF_freq.append(p)
#     else:
#         break

# hemp1=[]
# hemp2=[]
# for q in freqfunc: # 1-6
#     if q >=1 and q <= 6:
#         ner_freq.append(q)
#     elif q >=0 and q <= 1: #0-1
#         hemp1.append(q)
#     elif q > 6: #6 above
#         hemp2.append(q)
#     else:
#         break
#0-5 hz
# ampl = ampl[:len(FF_freq)]
# ampr = ampr[:len(FF_freq)]
# Nampl = Nampl[:len(FF_freq)]
# Nampr = Nampr[:len(FF_freq)]
#1-6
# Nampl = Nampl[len(hemp1):-len(hemp2)]
# Nampr = Nampr[len(hemp1):-len(hemp2)]

# print(len(FF_freq)) #512
# print(len(ner_freq)) #512
# print(len(hemp1)) #512
# print(len(hemp2)) #512
pp = int(len(freqfunc)/2)
print(pp)
freqfunc = freqfunc[:-pp]
ampl = ampl[:-pp]
ampr = ampr[:-pp]
Nampl = Nampl[:-pp]
Nampr = Nampr[:-pp]

#find_peak of amplitude
ampeaks_left, _ = find_peaks(ampl)
ampeaks_right, _ = find_peaks(ampr)
Nampeaks_left, _ = find_peaks(Nampl)
Nampeaks_right, _ = find_peaks(Nampr)

# print(len(ampeaks_left))#27
# print(len(ampeaks_right))#24
# print(len(Nampeaks_left))#29
# print(len(Nampeaks_right))#30

for a in ampeaks_left:
    peakleft.append( ampl [a] )
for b in ampeaks_right:
    peakright.append( ampr [b] )
for c in Nampeaks_left:
    Npeakleft.append( Nampl [c] )
for d in Nampeaks_right:
    Npeakright.append( Nampr [d] )


#find mean std and confidencial
mean_peakleft = s.mean(peakleft)
mean_peakright = s.mean(peakright)
Nmean_peakleft = s.mean(Npeakleft)
Nmean_peakright = s.mean(Npeakright)

std_peakleft = s.stdev(peakleft)
std_peakright = s.stdev(peakright)
Nstd_peakleft = s.stdev(Npeakleft)
Nstd_peakright = s.stdev(Npeakright)

sqrtSampleSize = (math.sqrt(delta_tXn))

templeft = std_peakleft/sqrtSampleSize
tempright = std_peakright/sqrtSampleSize
tempNleft = Nstd_peakleft/sqrtSampleSize
tempNright = Nstd_peakright/sqrtSampleSize

CI_left= stats.norm.interval(0.975, loc = mean_peakleft, scale = templeft)
lwb_left ,upb_left = CI_left
CI_right = stats.norm.interval(0.975, loc = mean_peakright, scale = tempright)
lwb_right ,upb_right = CI_right

CI_Nleft = stats.norm.interval(0.975, loc = Nmean_peakleft, scale = tempNleft)
lwb_Nleft ,upb_Nleft = CI_Nleft
CI_Nright = stats.norm.interval(0.975, loc = Nmean_peakright, scale = tempNright)
lwb_Nright ,upb_Nright = CI_Nright

print("Patient")
print('upper bound : ',upb_left)
print("mean :        ",mean_peakleft)
print('lower bound : ',lwb_left)
print("***********************")
print("Nervous")
print('upper bound : ',upb_Nleft)
print("mean  :       ",Nmean_peakleft)
print('lower bound : ',lwb_Nleft)

plt.subplot(2, 1, 1)
plt.plot(freqfunc,ampl, color='black')
#plt.plot(np.array(freqfunc)[l_amppeaks], np.array(ampl)[l_amppeaks], 'rx', label='Peaks in ampleft')
plt.axhline(y=upb_left, color='red',label = "amp upper bound")
plt.axhline(y=mean_peakleft, color='b',label = "mean")
plt.axhline(y=lwb_left, color='green',label = "amp lower bound")

plt.title("patient of Left Iris")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(freqfunc,Nampl, color='black')
#plt.plot(np.array(freqfunc)[r_amppeaks], np.array(ampr)[r_amppeaks], 'rx', label='Peaks in ampleft')
plt.axhline(y=upb_Nleft, color='red',label = "amp upper bound")
plt.axhline(y=Nmean_peakleft, color='b',label = "mean")
plt.axhline(y=lwb_Nleft, color='green',label = "amp lower bound")

plt.title("Nervous of Left Iris")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()