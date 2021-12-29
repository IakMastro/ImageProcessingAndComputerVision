import cv2
import numpy as np

from skimage.color import rgb2hsv
from skimage.segmentation import slic, mark_boundaries
from matplotlib import pyplot as plt


def plot_3d(x, y, z, file_name="rgb_3d", title='RGB Colors 3D', x_label='Blue', y_label='Green', z_label='Red') -> None:
  figure = plt.figure()
  axes = figure.add_subplot(111, projection='3d')
  axes.scatter(x, y, z)
  axes.set_title(title)
  axes.set_xlabel(x_label)
  axes.set_ylabel(y_label)
  axes.set_zlabel(z_label)

  plt.show()
  plt.savefig(f"{file_name}.png")


def plot(x, y, z, file_name="rgb_histogram", title='RGB Histogram', x_label='Blue', y_label='Green', z_label='Red') -> None:
  plt.plot(x, label=x_label)
  plt.plot(y, label=y_label, color='green')
  plt.plot(z, label=z_label, color='red')
  plt.title(title)
  plt.legend(loc='best')
  plt.show()
  plt.savefig(f"{file_name}.png")


def show_image(image, name) -> None:
  cv2.namedWindow(name, cv2.WINDOW_NORMAL)
  cv2.imshow(name, image)


def save_image(file_name, image) -> None:
  cv2.imwrite(f"{file_name}.png", 255 * image)


def plot_histogram(image, labels, bins, title, file_name, channels=('red', 'green', 'blue'), channel_ids=(0, 1, 2)) -> None:
  plt.xlim([0, 256])

  for label, channel_id, channel in zip(labels, channel_ids, channels):
    histogram, bin_edges = np.histogram(image[:, :, channel_id], bins=bins, range=(0, bins))
    plt.plot(bin_edges[0:-1], histogram, label=label, color=channel)

  plt.xlabel("Color value")
  plt.ylabel("Pixels")
  plt.title(title)
  plt.legend(loc="best")
  plt.show()
  plt.savefig(f"{file_name}.png")


if __name__ == '__main__':
  # Read OG image and convert it to RGB
  image = cv2.imread('the_doctor.jpg')
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # RGB's 3D plot and histogram
  # red_channel, green_channel, blue_channel = cv2.split(image)
  # plot_3d(blue_channel, green_channel, red_channel)
  # plot_histogram(image, ('Red', 'Green', 'Blue'), 256, "RGB Histogram", "rgb_histogram")

  # Convert RGB to HSV
  hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  save_image("hsv", hsv_image)

  # HSV's 3D plot and histogram
  # hue_channel, saturation_channel, value_channel = cv2.split(hsv_image)
  # plot_3d(hue_channel, saturation_channel, value_channel, "hsv_3d", 'HSV Colors', 'Hue', 'Saturation', 'Value')
  # plot_histogram(hsv_image, ('Hue', 'Saturation', 'Value'), 256, "Hue Histogram", "hue_histogram")

  # Combine RGB and HSV
  combined_image = np.concatenate((image, hsv_image), axis=2)
