import torch
import numpy as np
import pandas as pd
import os
import json
import torch.nn.functional as F
from torch.optim import Adam


# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directory where the task embedding files are located
directory_path = "/content/"

# List of all task embedding files
task_files = [
    "Assembly-v1.pth",
    "Basketball-v1.pth",
    "Bin-Picking-v1.pth",
    "Box-Close-v1.pth",
    "Button-Press-Topdown-v1.pth",
    "Button-Press-Topdown-Wall-v1.pth",
    "Button-Press-v1.pth",
    "Button-Press-Wall-v1.pth",
    "Coffee-Button-v1.pth",
    "Coffee-Pull-v1.pth",
    "Coffee-Push-v1.pth",
    "Dial-Turn-v1.pth",
    "Disassemble-v1.pth",
    "Door-Close-v1.pth",
    "Door-Lock-v1.pth",
    "Door-Open-v1.pth",
    "Door-Unlock-v1.pth",
    "Drawer-Close-v1.pth",
    "Drawer-Open-v1.pth",
    "Faucet-Close-v1.pth",
    "Faucet-Open-v1.pth",
    "Hammer-v1.pth",
    "Hand-Insert-v1.pth",
    "Handle-Press-Side-v1.pth",
    "Handle-Press-v1.pth",
    "Handle-Pull-Side-v1.pth",
    "Handle-Pull-v1.pth",
    "Lever-Pull-v1.pth",
    "Peg-Insert-Side-v1.pth",
    "Peg-Unplug-Side-v1.pth",
    "Pick-Out-Of-Hole-v1.pth",
    "Pick-Place-v1.pth",
    "Pick-Place-Wall-v1.pth",
    "Plate-Slide-Back-Side-v1.pth",
    "Plate-Slide-Back-v1.pth",
    "Plate-Slide-Side-v1.pth",
    "Plate-Slide-v1.pth",
    "Push-Back-v1.pth",
    "Push-v1.pth",
    "Push-Wall-v1.pth",
    "Reach-v1.pth",
    "Reach-Wall-v1.pth",
    "Shelf-Place-v1.pth",
    "Soccer-v1.pth",
    "Stick-Pull-v1.pth",
    "Stick-Push-v1.pth",
    "Sweep-Into-v1.pth",
    "Sweep-v1.pth",
    "Window-Close-v1.pth",
    "Window-Open-v1.pth",
]



tasks_to_remove = [
    "Assembly-v1.pth",
    "Basketball-v1.pth",
    "Bin-Picking-v1.pth",
    "Box-Close-v1.pth",
    "Button-Press-Topdown-v1.pth",
    "Button-Press-Topdown-Wall-v1.pth",
    "Button-Press-v1.pth",
    "Button-Press-Wall-v1.pth",
    "Coffee-Button-v1.pth",
    "Coffee-Pull-v1.pth",
    "Coffee-Push-v1.pth",
    "Dial-Turn-v1.pth",
    "Disassemble-v1.pth",
    "Door-Close-v1.pth",
    "Door-Lock-v1.pth",
    "Door-Open-v1.pth",
    "Door-Unlock-v1.pth",
    "Drawer-Close-v1.pth",
    "Drawer-Open-v1.pth",
    "Faucet-Close-v1.pth"]


filtered_task_files = [task_file for task_file in task_files if task_file not in tasks_to_remove]

# Verify if all files exist and print available files
available_files = [task_file for task_file in filtered_task_files if os.path.exists(os.path.join(directory_path, task_file))]
missing_files = [task_file for task_file in filtered_task_files if not os.path.exists(os.path.join(directory_path, task_file))]

print("Available files:", available_files)
print("Missing files:", missing_files)

# Load all available task embeddings
task_embeddings = [torch.load(os.path.join(directory_path, task_file), map_location=device) for task_file in available_files]

# Ensure all embeddings are tensors
task_embeddings = [torch.tensor(embedding, device=device) for embedding in task_embeddings]

# Stack all embeddings into a single tensor and remove duplicates
tensor_50 = torch.stack(task_embeddings)
unique_tensor_50, indices = torch.unique(tensor_50, dim=0, return_inverse=True)

# Create a dictionary to map unique tensor rows to task names
task_dict = {i: available_files[idx] for i, idx in enumerate(indices)}

# Convert back to tensor for further processing
tensor_50 = torch.tensor(unique_tensor_50, device=device)

# Function to calculate and print distance matrix using L-infinity norm
def calculate_distance_matrix_l_infinity(tensors):
    num_tensors = tensors.shape[0]
    distance_matrix = np.zeros((num_tensors, num_tensors))

    for i in range(num_tensors):
        for j in range(num_tensors):
            if i != j:
                distance_matrix[i, j] = torch.norm(tensors[i] - tensors[j], p=float('inf')).item()
            else:
                distance_matrix[i, j] = 0  # Distance to self is 0

    return distance_matrix

# Calculate the distance matrix
distance_matrix = calculate_distance_matrix_l_infinity(tensor_50)

# Convert the distance matrix to a DataFrame for better readability
distance_df = pd.DataFrame(distance_matrix, columns=available_files, index=available_files)

# Print the distance matrix
print("Distance Matrix:")
print(distance_df)

# Save the distance matrix to a CSV file for further inspection if needed
distance_df.to_csv(os.path.join(directory_path, 'distance_matrix_l_infinity.csv'), index=True)

# Save the unique tensor to a file
torch.save(tensor_50, os.path.join(directory_path, 'task_encoding.pth'))

# Save the task dictionary for future reference
with open(os.path.join(directory_path, 'task_dict.json'), 'w') as f:
    json.dump(task_dict, f)

print("Task Dictionary:")
print(task_dict)
# Save the task dictionary for future reference
import os
with open(os.path.join(directory_path, 'task_dict.json'), 'w') as f:
    json.dump(task_dict, f)

# Print the association of each task index with its task name
print("\nTask Index to Task Name Association:")
for index, task_name in task_dict.items():
    print(f"Task {index}: {task_name}")





# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tensor from a file and remove duplicates
tensor_50 = torch.load('task_encoding.pth', map_location=torch.device('cpu'))
tensor_50 = torch.unique(tensor_50, dim=0).cpu().numpy()  # Convert to NumPy array

# Parameters
K = 3
epsilon = .8

# Initialize cluster centers from specific indices of tensor_50
initial_indices = [0, 1, 2]  # Example indices from tensor_50 for initialization
initialization = torch.tensor(tensor_50[initial_indices], requires_grad=True, device=device)

# Initialize assignments
assignment = torch.zeros(len(tensor_50), K, device=device)
thetas_optimize = torch.tensor(tensor_50, device=device, dtype=torch.float32)

# Assign initial clusters based on L-infinity distance
for i in range(len(tensor_50)):
    for j in range(K):
        if torch.norm(thetas_optimize[i] - initialization[j], p=float('inf')) < epsilon:
            assignment[i][j] = 1

assignment = torch.tensor(assignment, requires_grad=True)

def softmax_coverage(thetas, thetas_optimized, alphas_optimized, epsilon, K, num_iterations=10000, lr=0.01):
    """Optimize the representative clusters using a softmax-based approach."""
    n = thetas.shape[0]

    # Initialize alphas
    alphas_optimized = torch.randn(n, K, requires_grad=True, device=device)

    optimizer = Adam([thetas_optimized, alphas_optimized], lr=lr)

    last_loss = 0
    loss = 999
    for j in range(num_iterations):
        if j % 1000 == 0:
            print(f"Loss {loss}")
        optimizer.zero_grad()
        loss = 0
        for i in range(n):
            sum_term = torch.sum(
                F.softmax(alphas_optimized[i], dim=0) * torch.stack(
                    [torch.norm(thetas_optimized[k] - thetas[i], p=float('inf')) for k in range(K)]
                )
            )
            loss += torch.relu(sum_term - epsilon)

        if abs(loss.item() - last_loss) < 1e-3 and j > 2000:
            break
        last_loss = loss.item()
        loss.backward()
        optimizer.step()

    return thetas_optimized, alphas_optimized

def assign_and_check_coverage(tensor_50, cluster_centers, epsilon):
    assignments = {i: [] for i in range(len(cluster_centers))}
    uncovered_tasks = []
    differences = []

    for i, task in enumerate(tensor_50):
        task_tensor = torch.tensor(task, device=device)
        min_distance = float('inf')

        for j, center in enumerate(cluster_centers):
            distance = torch.norm(task_tensor - center.clone().detach(), p=float('inf'))
            if not torch.isfinite(distance):
                print(f"Warning: Distance calculation resulted in inf or NaN for Task {i}, Center {j}")

            # Track minimum distance to any cluster center
            if distance < min_distance:
                min_distance = distance

        # Record the minimum distance for each task
        differences.append((i, min_distance))

        # Assign tasks to clusters if within epsilon
        for j, center in enumerate(cluster_centers):
            distance = torch.norm(task_tensor - center.clone().detach(), p=float('inf'))
            if distance <= epsilon:
                assignments[j].append((i, task_tensor.cpu().numpy()))

        if min_distance > epsilon:
            uncovered_tasks.append((i, task_tensor.cpu().numpy()))

    # Calculate covered tasks
    covered_tasks = []
    for key, value in assignments.items():
        for row_index, row_tensor in value:
            distance = torch.norm(torch.tensor(row_tensor, device=device) - cluster_centers[key].clone().detach(), p=float('inf')).item()
            if distance <= epsilon:
                covered_tasks.append((row_index, row_tensor))

    return covered_tasks, uncovered_tasks, differences, assignments

# Use softmax coverage
thetas_optimize = torch.tensor(tensor_50, device=device, dtype=torch.float32)
committee, assignment = softmax_coverage(thetas_optimize, initialization, assignment, epsilon, K)

# Verify coverage
covered_tasks_greedy, uncovered_tasks_greedy, differences_greedy, assignments_greedy = assign_and_check_coverage(tensor_50, committee, epsilon)

# Print results
print("\nCovered Tasks with Greedy Set Covering:")
for idx, tensor in covered_tasks_greedy:
    print(f"Task {idx} with tensor {tensor.tolist()} is covered.")

print("\nUncovered Tasks with Greedy Set Covering:")
for idx, tensor in uncovered_tasks_greedy:
    print(f"Task {idx} with tensor {tensor.tolist()} is NOT covered.")

print("\nDifferences from Cluster Centers:")
for idx, distance in differences_greedy:
    print(f"Task {idx} has a distance of {distance} from its nearest cluster center.")

# Print which tasks are in each cluster
print("\nTasks in each cluster:")
for cluster_id, tasks in assignments_greedy.items():
    task_ids = [task[0] for task in tasks]
    print(f"Cluster {cluster_id}: Tasks {task_ids}")


#Other clustering methods

#kmeans++

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tensor from a file and remove duplicates
tensor_50 = torch.load('task_encoding.pth', map_location=torch.device('cpu'))
tensor_50 = torch.unique(tensor_50, dim=0).cpu().numpy()  # Convert to NumPy array

# Check the shape of the tensor
print("Original shape of tensor:", tensor_50.shape)

# Use the tensor as-is for clustering
data = tensor_50

# Initialize variables
n_samples = data.shape[0]
n_features = data.shape[1:]  # This allows handling of the multidimensional nature of the data
k = 3  # Number of clusters
max_iters = 100  # Maximum number of iterations
tolerance = 1e-4  # Convergence tolerance

def calculate_distance(data_point, centroid):
    """Calculate the L-infinity distance between a data point and a centroid in multidimensional space."""
    return np.max(np.abs(data_point - centroid))

def initialize_centroids_kmeanspp(data, k):
    """Initialize centroids using the k-means++ algorithm."""
    np.random.seed(0)  # For reproducibility
    centroids = []
    # Choose the first centroid randomly
    first_index = np.random.choice(n_samples)
    centroids.append(data[first_index])

    # Choose the remaining centroids
    for _ in range(1, k):
        # Compute the squared distance from each point to its nearest centroid
        distances = np.array([
            min([calculate_distance(point, centroid)**2 for centroid in centroids])
            for point in data
        ])
        # Compute probabilities proportional to the squared distances
        probabilities = distances / distances.sum()
        # Choose a new centroid with the computed probability distribution
        new_index = np.random.choice(n_samples, p=probabilities)
        centroids.append(data[new_index])

    return np.array(centroids)

# Initialize centroids using k-means++
centroids = initialize_centroids_kmeanspp(data, k)

def assign_clusters(data, centroids):
    """Assign each data point to the nearest centroid."""
    labels = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        distances = [calculate_distance(data[i], centroid) for centroid in centroids]
        labels[i] = np.argmin(distances)
    return labels

def update_centroids(data, labels, k):
    """Update centroids as the mean of assigned points, considering multidimensionality."""
    new_centroids = np.zeros((k, *n_features))
    for i in range(k):
        assigned_points = data[labels == i]
        if len(assigned_points) > 0:
            new_centroids[i] = assigned_points.mean(axis=0)
    return new_centroids

# KMeans algorithm
for iteration in range(max_iters):
    # Step 1: Assign clusters
    labels = assign_clusters(data, centroids)

    # Step 2: Update centroids
    new_centroids = update_centroids(data, labels, k)

    # Check for convergence
    if np.all(np.abs(new_centroids - centroids) < tolerance):
        print(f"Converged in {iteration} iterations.")
        break

    centroids = new_centroids

# Final cluster assignment based on epsilon ball
epsilon = 1  # Set epsilon for coverage check

def assign_tasks_within_epsilon(data, centroids, epsilon):
    """Assign tasks to clusters based on epsilon distance."""
    clusters_with_epsilon = {i: [] for i in range(k)}
    for i, point in enumerate(data):
        for j, centroid in enumerate(centroids):
            distance = calculate_distance(point, centroid)
            if distance <= epsilon:
                clusters_with_epsilon[j].append(i)
    return clusters_with_epsilon

clusters_with_epsilon = assign_tasks_within_epsilon(data, centroids, epsilon)

# Print results
print("\nFinal cluster centroids:")
print(centroids)

print("\nCluster assignments for each task (based on epsilon distance):")
for cluster_id, task_ids in clusters_with_epsilon.items():
    print(f"Cluster {cluster_id}: Tasks {task_ids}")

def check_coverage(data, centroids, clusters_with_epsilon, epsilon):
    """Verify cluster coverage based on epsilon."""
    covered_tasks = []
    uncovered_tasks = []
    differences = []

    for i, point in enumerate(data):
        is_covered = False
        for cluster_id, task_ids in clusters_with_epsilon.items():
            if i in task_ids:
                centroid = centroids[cluster_id]
                distance = calculate_distance(point, centroid)
                differences.append((i, distance))
                if distance <= epsilon:
                    covered_tasks.append(i)
                    is_covered = True
                    break
        if not is_covered:
            uncovered_tasks.append(i)

    return covered_tasks, uncovered_tasks, differences

covered_tasks, uncovered_tasks, differences = check_coverage(data, centroids, clusters_with_epsilon, epsilon)

print("\nCovered Tasks:")
for idx in covered_tasks:
    print(f"Task {idx} is within {epsilon} distance of its centroid.")

print("\nUncovered Tasks:")
for idx in uncovered_tasks:
    print(f"Task {idx} is not within {epsilon} distance of any centroid.")

print("\nDifferences from Cluster Centers:")
for idx, distance in differences:
    print(f"Task {idx} has a distance of {distance} from its nearest cluster center.")


#DBScan
import torch
import numpy as np

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tensor from a file and remove duplicates
tensor_10 = torch.load('task_encoding.pth', map_location=torch.device('cpu'))
tensor_10 = torch.unique(tensor_10, dim=0).cpu().numpy()  # Convert to NumPy array

# Check the shape of the tensor
print("Original shape of tensor:", tensor_10.shape)

# Use the tensor as-is for clustering
data = tensor_10
n_samples = data.shape[0]

# DBScan parameters
epsilon = .74    # Radius for neighborhood search
min_pts = 2         # Minimum number of points to form a dense region

def calculate_distance(data_point, centroid):
    """Calculate the L-infinity (Chebyshev) distance between two points."""
    return np.max(np.abs(data_point - centroid))

def region_query(data, point_index, epsilon):
    """
    Return indices of all points within epsilon distance
    from the data point at point_index.
    """
    neighbors = []
    point = data[point_index]
    for i in range(n_samples):
        distance = calculate_distance(data[i], point)
        if distance <= epsilon:
            neighbors.append(i)
    return neighbors

def expand_cluster(data, labels, point_index, cluster_id, epsilon, min_pts, visited):
    """
    Expand the cluster recursively.
    If the current point has fewer than min_pts neighbors, mark it as noise.
    Otherwise, add all its density-reachable points to the cluster.
    """
    neighbors = region_query(data, point_index, epsilon)

    if len(neighbors) < min_pts:
        labels[point_index] = -1  # Mark as noise
        return False
    else:
        # Assign the current point to the cluster
        labels[point_index] = cluster_id

        # Initialize seeds with all neighbors (excluding the current point)
        seeds = list(neighbors)
        if point_index in seeds:
            seeds.remove(point_index)

        while seeds:
            current_point = seeds.pop(0)
            if not visited[current_point]:
                visited[current_point] = True
                current_neighbors = region_query(data, current_point, epsilon)
                if len(current_neighbors) >= min_pts:
                    for n in current_neighbors:
                        if n not in seeds:
                            seeds.append(n)
            # If current point is not yet assigned (or was marked as noise), assign it to cluster
            if labels[current_point] in [0, -1]:
                labels[current_point] = cluster_id
        return True

def dbscan(data, epsilon, min_pts):
    """
    DBScan clustering algorithm.

    Returns:
        labels: A numpy array of cluster labels for each data point.
                Noise points are labeled as -1.
    """
    labels = np.zeros(n_samples, dtype=int)  # 0 means unclassified
    visited = np.zeros(n_samples, dtype=bool)
    cluster_id = 1

    for i in range(n_samples):
        if not visited[i]:
            visited[i] = True
            if expand_cluster(data, labels, i, cluster_id, epsilon, min_pts, visited):
                cluster_id += 1
    return labels

# Run DBScan on data
labels = dbscan(data, epsilon, min_pts)

# Compute cluster centroids as the mean of points in each non-noise cluster.
unique_clusters = np.unique(labels)
unique_clusters = unique_clusters[unique_clusters != -1]  # Remove noise (-1)
centroids = []
for cluster in unique_clusters:
    cluster_points = data[labels == cluster]
    centroid = cluster_points.mean(axis=0)
    centroids.append(centroid)
centroids = np.array(centroids)

# Create a dictionary mapping each cluster to the list of task indices.
clusters_with_epsilon = {}
for cluster in unique_clusters:
    clusters_with_epsilon[cluster] = list(np.where(labels == cluster)[0])

# Print the results
print("\nFinal cluster centroids:")
print(centroids)

print("\nCluster assignments for each task (based on DBScan):")
for cluster_id, task_ids in clusters_with_epsilon.items():
    print(f"Cluster {cluster_id}: Tasks {task_ids}")

def check_coverage(data, centroids, clusters_with_epsilon, epsilon):
    """
    Verify cluster coverage.

    For each data point, compute the distance to its nearest cluster centroid.
    Return lists of covered and uncovered tasks, and a list of (task_index, distance) pairs.
    """
    covered_tasks = []
    uncovered_tasks = []
    differences = []

    for i, point in enumerate(data):
        min_distance = np.inf
        for cluster_id, centroid in zip(clusters_with_epsilon.keys(), centroids):
            distance = calculate_distance(point, centroid)
            if distance < min_distance:
                min_distance = distance
        differences.append((i, min_distance))
        if min_distance <= epsilon:
            covered_tasks.append(i)
        else:
            uncovered_tasks.append(i)

    return covered_tasks, uncovered_tasks, differences

covered_tasks, uncovered_tasks, differences = check_coverage(data, centroids, clusters_with_epsilon, epsilon)

print("\nCovered Tasks:")
for idx in covered_tasks:
    print(f"Task {idx} is within {epsilon} distance of its centroid.")

print("\nUncovered Tasks:")
for idx in uncovered_tasks:
    print(f"Task {idx} is not within {epsilon} distance of any centroid.")

print("\nDifferences from Cluster Centers:")
for idx, distance in differences:
    print(f"Task {idx} has a distance of {distance} from its nearest cluster center.")



#GMM
import torch
import numpy as np
from sklearn.mixture import GaussianMixture

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tensor from a file and remove duplicates
tensor_10 = torch.load('task_encoding.pth', map_location=torch.device('cpu'))
tensor_10 = torch.unique(tensor_10, dim=0).cpu().numpy()  # Convert to NumPy array

# Check the shape of the tensor
print("Original shape of tensor:", tensor_10.shape)

# If the tensor has more than 2 dimensions, flatten it to shape (n_samples, -1)
if tensor_10.ndim > 2:
    data = tensor_10.reshape(tensor_10.shape[0], -1)
else:
    data = tensor_10

n_samples = data.shape[0]

# Set parameters for GMM clustering
k = 3         # Number of clusters/components
max_iters = 100  # Maximum number of EM iterations
tolerance = 1e-4 # Convergence tolerance for the EM algorithm
epsilon = 0.6    # Threshold distance for coverage check

def calculate_distance(data_point, centroid):
    """Calculate the L-infinity (Chebyshev) distance between a data point and a centroid."""
    return np.max(np.abs(data_point - centroid))

# --- Gaussian Mixture Clustering ---
# Initialize and fit the Gaussian Mixture Model
gmm = GaussianMixture(n_components=k, max_iter=max_iters, tol=tolerance, random_state=0)
gmm.fit(data)

# Obtain the cluster assignments for each data point
labels = gmm.predict(data)

# The centroids for each Gaussian component are given by the means
centroids = gmm.means_

# Create a dictionary mapping each cluster to the list of data point indices
clusters = {i: [] for i in range(k)}
for i, label in enumerate(labels):
    clusters[label].append(i)

# Print the clustering results
print("\nFinal cluster centroids (GMM means):")
print(centroids)

print("\nCluster assignments for each task (based on GMM):")
for cluster_id, task_ids in clusters.items():
    print(f"Cluster {cluster_id}: Tasks {task_ids}")

def check_coverage(data, centroids, epsilon):
    """
    Verify cluster coverage based on epsilon.

    For each data point, compute the L-infinity distance to its nearest cluster centroid.
    Return lists of covered and uncovered tasks, and a list of (data_index, distance) pairs.
    """
    covered_tasks = []
    uncovered_tasks = []
    differences = []

    for i, point in enumerate(data):
        min_distance = np.inf
        for centroid in centroids:
            distance = calculate_distance(point, centroid)
            if distance < min_distance:
                min_distance = distance
        differences.append((i, min_distance))
        if min_distance <= 999:
            covered_tasks.append(i)
        else:
            uncovered_tasks.append(i)

    return covered_tasks, uncovered_tasks, differences

# Evaluate cluster coverage: which tasks are within epsilon distance of a centroid
covered_tasks, uncovered_tasks, differences = check_coverage(data, centroids, epsilon)

print("\nCovered Tasks:")
for idx in covered_tasks:
    print(f"Task {idx} is within {epsilon} distance of its nearest centroid.")

print("\nUncovered Tasks:")
for idx in uncovered_tasks:
    print(f"Task {idx} is not within {epsilon} distance of any centroid.")

print("\nDifferences from Cluster Centers:")
for idx, distance in differences:
    print(f"Task {idx} has a distance of {distance} from its nearest cluster center.")

