import cv2
import numpy as np

from matplotlib import pyplot as plt


# Canny function
def apply_edges(image, t_lower=100, t_upper=200, aperture_size=5, L2Gradient=True) -> any:
  return cv2.Canny(image, t_lower, t_upper, apertureSize=aperture_size, L2gradient=L2Gradient)


# Noise and removal function
def add_noise_and_remove_it(image) -> any:
  gaussian_noise = np.random.normal(0, 1, image.size)
  gaussian_noise = gaussian_noise.reshape(image.shape[0], image.shape[1]).astype('uint8')
  noise = cv2.add(image, gaussian_noise)
  return cv2.medianBlur(noise, 5)


# Check similarity function
def check_similarity(image, image_with_edges, name) -> None:
  # Histogram calculation
  histogram_original = cv2.calcHist([image], [0], None, [256], [0, 256])
  histogram_edges = cv2.calcHist([image_with_edges], [0], None, [256], [0, 256])
  histogram_original = cv2.normalize(histogram_original, histogram_original).flatten()
  histogram_edges = cv2.normalize(histogram_edges, histogram_edges).flatten()

  # Euclidean distance calculation
  euclidean = 0
  i = 0
  while i < len(histogram_original) and i < len(histogram_edges):
    euclidean += (histogram_original[i] - histogram_edges[i]) ** 2
    i += 1

  # Printing the distance with various comparissons
  print(f"\tEuclidean distance: {euclidean ** (1 / 2)}")
  print(f"\tChi-Squared distance: {cv2.compareHist(histogram_original, histogram_edges, cv2.HISTCMP_CHISQR)}")
  print(f"\tBhattacharyya distance: {cv2.compareHist(histogram_original, histogram_edges, cv2.HISTCMP_BHATTACHARYYA)}")
  print(f"\tCorrelation distance: {cv2.compareHist(histogram_original, histogram_edges, cv2.HISTCMP_CORREL)}")
  print(f"\tIntersection distance: {cv2.compareHist(histogram_original, histogram_edges, cv2.HISTCMP_INTERSECT)}")

  # Plotting the differences
  plt.figure(name)
  plt.subplot(1, 2, 1)
  plt.imshow(image)
  plt.subplot(1, 2, 2)
  plt.imshow(image_with_edges)
  plt.savefig(f"{name}.png")
  plt.show()


if __name__ == '__main__':
  image = cv2.imread('david_gilmour.jpg', 0)

  print("Original image:")
  check_similarity(image, apply_edges(image), "Original")

  noisy_image = add_noise_and_remove_it(image)

  print("Restored image:")
  check_similarity(noisy_image, apply_edges(noisy_image), "Restored")

  kernel = np.ones((5,5), np.uint8)
  erosion = cv2.erode(noisy_image, kernel=kernel, iterations=1)

  print("Erosion Morphology:")
  check_similarity(erosion, apply_edges(erosion), "Erosion")

  dilation = cv2.dilate(noisy_image, kernel=kernel, iterations=1)

  print("Dilation Morphology:")
  check_similarity(dilation, apply_edges(dilation), "Dilation")
