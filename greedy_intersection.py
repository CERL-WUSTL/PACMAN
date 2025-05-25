import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Parameters for the three modes
model1_weight=0.4
mode1_mean = np.array([0.25, 0.25, 0.25])
mode1_cov = 0.05*np.array([[1, 0.5, 0.5],
                       [0.5, 1, 0.5],
                       [0.5, 0.5, 1]])
model2_weight=0.4
mode2_mean = np.array([0.5, 0.15, 0.15])
mode2_cov = 0.05*np.array([[1, -0.5, -0.5],
                       [-0.5, 1, -0.5],
                       [-0.5, -0.5, 1]])
model3_weight=0.2
mode3_mean = np.array([0.6, 0.15, 0.1])
mode3_cov = 0.05*np.array([[1, 0, 0.5],
                       [0, 1, 0],
                       [0.5, 0, 1]])
# Generate random samples for each mode
mode1_samples = np.random.multivariate_normal(mode1_mean, mode1_cov, 40)
mode2_samples = np.random.multivariate_normal(mode2_mean, mode2_cov, 40)
mode3_samples = np.random.multivariate_normal(mode3_mean, mode3_cov, 20)
# Combine samples from different modes
data = np.vstack((mode1_samples, mode2_samples, mode3_samples))
# Visualize the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
# Initialize the list with the correct number of elements
thetas_optimize = [None] * len(data)

# Populate the list with the tensors
for i, (x,y,z) in enumerate(data):
  a = torch.tensor([x,y,z, 1 - (x+y+z)])
  thetas_optimize[i] = a

# Print the list
print(thetas_optimize)
