import numpy as np
import cv2


# Solarize function
def solarize(image, threshold):
    return np.where((image < threshold), image, ~image)


def show_image(image, name, filename):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.imwrite(f'{filename}.png', 255 * image)


if __name__ == '__main__':
    image = cv2.imread('zelda.jpeg', 0)
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", image)

    show_image(solarize(image, 64), "Threshold 64", "threshold_64")
    show_image(solarize(image, 128), "Threshold 128", "threshold_128")
    show_image(solarize(image, 192), "Threshold 192", "threshold_192")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

