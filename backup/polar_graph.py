import math
import numpy as np
import matplotlib.pyplot as plt

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
path = "C:/Users/pnaSu/Desktop/openCV_project/csv/"
file1 = open(path+'leftcenteriris_lefteye.csv', 'r') 
file2 = open(path+'leftcenteriris_righteye.csv', 'r') 
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
        # print(r_iris_anglesx)
        # print(r_iris_anglesy)
        # print(r_iris_radius)
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
        tempr = (math.atan2(delta_right_iris_y[i], delta_right_iris_x[i]))
        #print(templ)
        l_angular_movement.append(templ)
        r_angular_movement.append(tempr)
        i+=1
l_angular_movement_radians = np.radians(l_angular_movement)
r_angular_movement_radians = np.radians(r_angular_movement)

# Plot polar graph for the left iris
plt.subplot(2, 1, 1, projection='polar')
plt.scatter(np.linspace(0, 2*np.pi, len(l_angular_movement_radians)), l_angular_movement_radians)
plt.title('Angular Movement of Left Iris (Polar Plot)')

# # Plot polar graph for the right iris
# plt.subplot(2, 1, 2, projection='polar')
# plt.plot(np.linspace(0, 2*np.pi, len(r_angular_movement_radians)), r_angular_movement_radians)
# plt.title('Angular Movement of Right Iris (Polar Plot)')

plt.show()
