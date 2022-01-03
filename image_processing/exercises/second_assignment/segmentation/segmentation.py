import cv2
import numpy as np

from skimage.color import label2rgb
from sklearn.cluster import MeanShift, estimate_bandwidth, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt


# 3D plots of the channels
def plot_3d(x, y, z, filename="rgb_3d", title='RGB Colors 3D', x_label='Blue', y_label='Green', z_label='Red') -> None:
  figure = plt.figure()
  axes = figure.add_subplot(111, projection='3d')
  axes.scatter(x, y, z)
  axes.set_title(title)
  axes.set_xlabel(x_label)
  axes.set_ylabel(y_label)
  axes.set_zlabel(z_label)

  plt.savefig(f"{filename}.png")
  plt.show()


# Plot an image
def plot(filename, title, image):
  plt.imshow(image)
  plt.title(title)
  plt.savefig(f"{filename}")
  plt.show()


# Save the image
def save_image(filename, image) -> None:
  cv2.imwrite(f"{filename}.png", 255 * image)


# Ploting the histogram
def plot_histogram(image, labels, bins, title, filename, channels=('red', 'green', 'blue'), channel_ids=(0, 1, 2)) -> None:
  plt.xlim([0, 256])

  # Histogram calculation on each channel
  for label, channel_id, channel in zip(labels, channel_ids, channels):
    histogram, bin_edges = np.histogram(image[:, :, channel_id], bins=bins, range=(0, bins))
    plt.plot(bin_edges[0:-1], histogram, label=label, color=channel)

  plt.xlabel("Color value")
  plt.ylabel("Pixels")
  plt.title(title)
  plt.legend(loc="best")
  plt.savefig(f"{filename}.png")
  plt.show()


# Segmentation algorithm
def segmentation(image, shape, color_code, channel_amount, algorithm, algorithm_name):
  labels = algorithm.labels_
  labels_unique = np.unique(labels)
  print(f"{algorithm_name}:\n\tChannels: {channel_amount}\n\tColor mode: {color_code}")
  n_clusters = len(labels_unique)
  print(f"\tClusters: {n_clusters}")
  segmented_image = (label2rgb(np.reshape(labels, shape[:2]), bg_label=0) * 255).astype(np.uint8)
  # plot(f"{color_code}_{algorithm_name}", f"{color_code} {algorithm_name}", segmented_image)
  print(f"\tSilhouette Score: {silhouette_score(image, labels, sample_size=25000)}")


def segment_image(image, color_code, channel_amount=3) -> None:
  shape = image.shape
  flat_image = np.reshape(image, [-1, channel_amount])

  # MeanShift segmentation
  bandwidth = estimate_bandwidth(flat_image, quantile=0.1, n_samples=100)
  meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
  meanshift.fit(flat_image)
  segmentation(flat_image, shape, color_code, channel_amount, meanshift, "MeanShift")

  # KMeans segmentation
  k_means = MiniBatchKMeans()
  k_means.fit(flat_image)
  segmentation(flat_image, shape, color_code, channel_amount, k_means, "KMeans")
  print("\n===========================")


if __name__ == '__main__':
  # Read OG image and convert it to RGB
  image = cv2.imread('the_doctor.jpg')
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # RGB's 3D plot and histogram
  red_channel, green_channel, blue_channel = cv2.split(image)
  # plot_3d(blue_channel, green_channel, red_channel)
  # plot_histogram(image, ('Red', 'Green', 'Blue'), 256, "RGB Histogram", "rgb_histogram")
  segment_image(image, "RGB")

  # Convert RGB to HSV
  hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  save_image("hsv", hsv_image)

  # HSV's 3D plot and histogram
  hue_channel, saturation_channel, value_channel = cv2.split(hsv_image)
  # plot_3d(hue_channel, saturation_channel, value_channel, "hsv_3d", 'HSV Colors', 'Hue', 'Saturation', 'Value')
  # plot_histogram(hsv_image, ('Hue', 'Saturation', 'Value'), 256, "Hue Histogram", "hue_histogram")
  segment_image(hsv_image, "HSV")

  # Combine RGB and HSV
  combined_image = np.concatenate((image, hsv_image), axis=2)
  segment_image(combined_image, "RGB&HSV", 6)
