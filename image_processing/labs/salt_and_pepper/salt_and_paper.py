import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim
from random import random


def show_image(og_image, image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)

    simil_score, _ = ssim(og_image, image, full=True)
    print(f"{name} SSIM score is: {simil_score}")


def salt_and_paper(image, probability):
    noise_image = np.zeros(image.shape, np.uint8)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            rand = random()
            if rand < probability:
                noise_image[row][col] = 0
            elif rand > (1 - probability):
                noise_image[row][col] = 255
            else:
                noise_image[row][col] = image[row][col]
    
    return noise_image


if __name__ == '__main__':
    image = cv2.imread('img/face_value.jpg', 0)

    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Original Image', image)

    kernel = np.ones((5, 5), np.float32) / 25

    average_image = cv2.filter2D(image, -1, kernel)
    gauss_image = cv2.GaussianBlur(image, (5, 5), 0)

    show_image(image, average_image, 'Average Image')
    show_image(image, gauss_image, 'Gaus Image')

    kernel = np.array(
        [
            [0, -1 , 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]
    )

    sharp_image = cv2.filter2D(image, -1, kernel)
    sharp_gauss_image = cv2.filter2D(gauss_image, -1, kernel)

    show_image(image, sharp_image, 'Sharp Image')
    show_image(image, sharp_gauss_image, 'Sharp Gaus Image')

    noise_image = salt_and_paper(image, 0.12)
    show_image(image, noise_image, 'Noise Image')

    median = cv2.medianBlur(noise_image, 5)
    show_image(image, median, 'Median')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
