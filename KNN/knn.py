import math

def euclidean_distance(p1, p2):
  """Calculates the Euclidean distance between two data points."""
  distance = 0
  for i in range(len(p1)):
    distance += (p1[i] - p2[i])**2
  return math.sqrt(distance)

def knn_predict(data, target, new_point, k):
  """Predicts the class label for a new data point using KNN."""
  distances = []
  for i in range(len(data)):
    distance = euclidean_distance(data[i], new_point)
    distances.append((distance, target[i]))  # Store distance and target label

  # Sort distances based on ascending distance
  distances.sort(key=lambda x: x[0])

  # Get k nearest neighbors' labels
  k_nearest_labels = [label for distance, label in distances[:k]]

  # Determine most frequent class label among k neighbors
  from collections import Counter
  label_counts = Counter(k_nearest_labels)
  most_frequent_label = label_counts.most_common(1)[0][0]

  return most_frequent_label

# Example usage (assuming you have your data and target labels)
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
target = ["A", "B", "A"]
new_point = [5, 4, 7]
k = 3

predicted_label = knn_predict(data, target, new_point, k)
print("Predicted label for the new point:", predicted_label)
