import matplotlib.pyplot as plt

# define the two sets of coordinates
#coordinates1 = [(701.1989180857236, 366.7588386092959), (1073.1370849898476, 804.4579361637689), (751.4034995499685, 502.52334059711313), (540.3321253657839, 303.47278169309993), (893.5319625684647, 696.9777054234136), (1031.7713382904349, 876.9363812353898), (844.3880412759995, 812.5896641474142)]
coordinates2 = [[34.5, 12.5], [38.5, 12.5], [40.5, 17], [44, 24], [44.5, 20], [48, 24], [49, 20]]



# separate x and y values for each set of coordinates
#x1, y1 = zip(*coordinates1)
x2, y2 = zip(*coordinates2)

# plot the first set of coordinates in blue
#plt.plot(x1, y1, 'bo', label='Coordinates 1')

# plot the second set of coordinates in red
plt.plot(x2, y2, 'ro', label='Coordinates 2')

# add legend, x and y labels, and title
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of Two Sets of Coordinates')

# show the plot
plt.show()