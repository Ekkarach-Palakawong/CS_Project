import numpy as np
import matplotlib.pyplot as plt

iris_angles = []
iris_radius = []
i=0
file1 = open('test_right.csv', 'r') 
try:
        for line in file1:
            line_values = line.split(',')
            angle = float(line_values[0])
            radius = float(line_values[1])

            iris_angles.append(angle)
            iris_radius.append(radius)

except FileExistsError as e:
        print(e)
except Exception as e:
        print(e)
else:
        file1.close()

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='polar')


ax.scatter(iris_angles, iris_radius, c='b', s=50, alpha=0.5)

ax.set_rlabel_position(90)  # Rotate radial labels to the top
ax.set_rticks([])           # Remove radial tick labels
ax.grid(True)

plt.title("Iris Position (Polar Plot)")

plt.show()