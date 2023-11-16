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

delta_left_iris_x = []
delta_left_iris_y = []
delta_right_iris_x = []
delta_right_iris_y = []

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
        templ = math.degrees(math.atan2( delta_left_iris_y[i],delta_left_iris_x[i])) #return radians
        tempr = math.degrees(math.atan2(delta_right_iris_y[i], delta_right_iris_x[i]))
        #print(templ)
        l_angular_movement.append(templ)
        r_angular_movement.append(tempr)
        i+=1
frame_indices1 = range(len(l_angular_movement))
frame_indices2 = range(len(r_angular_movement))
#ax.scatter(iris_anglesx, iris_radius, c='b', s=50, alpha=0.5)

# ax.set_rlabel_position(90)  # Rotate radial labels to the top
# ax.set_rticks([])           # Remove radial tick labels

# plt.figure(figsize=(8, 8))
# plt.subplot(111, projection='polar')

# plt.plot(angular_movement_x,label = 'x')
# plt.plot(angular_movement_y,label = 'y')
plt.plot(frame_indices1, l_angular_movement)
plt.plot(frame_indices2, r_angular_movement)
plt.title('Angular Movement of Left Iris on X-axis')
plt.xlabel('Frame')
plt.ylabel('Angular Movement (degrees)')
plt.grid(True)
plt.show()