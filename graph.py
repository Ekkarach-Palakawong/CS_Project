import matplotlib.pyplot as plt

x = []
y = []
arrow_points = []
i=0
file1 = open('test_right.csv', 'r') 
try:
        for line in file1:
                line_values = line.split(',')

                x_value = float(line_values[0])
                y_value = float(line_values[1])
                
                x.append(x_value)
                y.append(y_value)

        plt.plot(x, y)

        while i<len(x):
                if i==(len(x)-1):
                        break
                else:
                        arrow_points.append(
                                (x[i],y[i] ,x[i+1],y[i+1])
                        )
                        i+=1
        for start_x, start_y, end_x, end_y in arrow_points:
                plt.annotate(
                        '',
                        (end_x, end_y),  # End point of the arrow
                        (start_x, start_y),  # Start point of the arrow
                        arrowprops=dict(arrowstyle='->', color='red', linewidth=2),
                        annotation_clip=False  # This allows the annotation to be outside the plot area
                )
        
except FileNotFoundError as e:
        print(e)
except Exception as e:
        print(e)
else:
        file1.close()
        
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot from......csv')

plt.show()