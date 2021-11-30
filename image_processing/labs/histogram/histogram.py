import cv2
import numpy as np


def show_image(image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)    


def create_histogram(image):
    image_normalized_histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    return image_normalized_histogram / sum(image_normalized_histogram)


def show_plots(blue_histogram, green_histogram, red_histogram):
    from matplotlib import pyplot as plt

    plt.plot(blue_histogram, label="Blue")
    plt.plot(green_histogram, label="Green", color="green")
    plt.plot(red_histogram, label="Red", color='red')
    plt.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    image = cv2.imread('img/foxtrot.jpg')

    blue_channel, green_channel, red_channel = cv2.split(image)

    show_image(blue_channel, "Blue Channel")
    show_image(green_channel, "Green Channel")
    show_image(red_channel, "Red Channel")

    cv2.waitKey()
    cv2.destroyAllWindows()

    blue_histogram = create_histogram(blue_channel)
    green_histogram = create_histogram(green_channel)
    red_histogram = create_histogram(red_channel)

    show_plots(blue_histogram, green_histogram, red_histogram)

    blue_equalized = cv2.equalizeHist(blue_channel)
    green_equalized = cv2.equalizeHist(green_channel)
    red_equalized = cv2.equalizeHist(red_channel)

    blue_equalized_histogram = create_histogram(blue_equalized)
    green_equalized_histogram = create_histogram(green_equalized)
    red_equalized_histogram = create_histogram(red_equalized)

    show_plots(blue_equalized_histogram, green_equalized_histogram, red_equalized_histogram)

    synthesized_image = cv2.merge((blue_equalized, green_equalized, red_equalized))

    show_image(synthesized_image, "Merged Image")
    cv2.waitKey()
    cv2.destroyAllWindows()
