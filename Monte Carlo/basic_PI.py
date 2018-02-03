import numpy as np
import matplotlib.pyplot as plt

#number of samples
N_total = 10_000

#drawing random points uniform between -1 and 1
X = np.random.uniform(low=-1, high=1, size=N_total)  
Y = np.random.uniform(low=-1, high=1, size=N_total)   

# calculate the distance of the points from the center 
distance = np.sqrt(X**2+Y**2);
# check if point is inside the circle
is_point_inside = distance<1.0
# sum up the hits inside the circle
N_inside=np.sum(is_point_inside)

#area of the bounding box [-1 -> 1 = 2]^2
box_area = 4.0
# estimate the circle area
circle_area = box_area * N_inside/N_total

# results
print("Area of the circle = ", circle_area)
print("pi = ", np.pi)
print

# plot
plt.scatter(X,Y, c=is_point_inside, alpha=0.5, cmap=plt.cm.Set3)
plt.axis('equal')
plt.axhline(0, color='white')
plt.axvline(0, color='white')
# plt.show()
