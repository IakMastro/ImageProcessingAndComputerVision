import cv2
import numpy as np

def gamma_correction(image, gamma):
    inverted_gamma = 1 / gamma
    
    table = [((i / 255) ** inverted_gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(image, table)


if __name__ == '__main__':
    image = cv2.imread('daleks.webp')
    gamma_image = gamma_correction(image, 2.2)

    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Original Image', image)

    cv2.namedWindow('Gamma Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Gamma Image', gamma_image)

    cv2.waitKey()
    cv2.destroyAllWindows()