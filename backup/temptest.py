from scipy.fft import fft
import csv

ampleftx = []
amplefty = []
amprightx = []
amprighty = []
freqfunc = [0]

peakleftx = []
peaklefty = []
peakrightx = []
peakrighty = []
normal_peakleftx = []
normal_peaklefty = []
normal_peakrightx = []
normal_peakrighty = []

Nampleftx = []
Namplefty = []
Namprightx = []
Namprighty = []

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
file3 = open(path+'Nervous2centeriris_lefteye.csv', 'r')
file4 = open(path+'Nervous2centeriris_righteye.csv', 'r')

filex1 = open('temp1.csv', 'w')
filex2 = open('temp2.csv', 'w')
writer1 = csv.writer(filex1)
writer2 = csv.writer(filex2)
t = 0
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

    N_fftleftx = fft(normalleftx)

    fft_leftx = fft(left_anglex)


    while t < len(fft_leftx):
        writer1.writerow(fft_leftx)
        writer1.writerow(N_fftleftx)
        t += 1
except FileExistsError as e:
    print(e)
except Exception as e:
    print(e)
else:
    file1.close()
    file2.close()
    file3.close()
    file4.close()

n = 256 #sample size 256 128
delta_t = 1/30 #sampling time
delta_tXn = delta_t*n # this is used to find a freq
