import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim
from random import random


def show_image(image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)


# Calculate Structural Similarity Index
def calc_simil(name, original_image, image):
    simil_score, _ = ssim(original_image, image, full=True)
    print(f"{name} SSIM score is: {simil_score}")


# Calculate Mean Squared Error
def calc_mean(name, original_image, image):
    mean_squared_error = np.square(np.subtract(original_image, image)).mean()
    print(f"{name} Mean Squared Error is: {mean_squared_error}")


# Remove noise function
def remove_noise(noise_name, file_name, noise_image, original_image):
    # Guassian  blur
    kernel = np.ones((5, 5), np.float32) / 25
    gauss_image = cv2.GaussianBlur(noise_image, (5, 5), 0)

    # Guassian sharp
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]
    )

    sharp_gauss = cv2.filter2D(gauss_image, -1, kernel)
    show_image(gauss_image, f"{noise_name} Gaussian Sharp")
    cv2.imwrite(f"{file_name}_gauss_sharp.png", 255 * sharp_gauss)
    calc_simil(f"{noise_name} Guassian Sharp without Noise", original_image, sharp_gauss)
    calc_mean(f"{noise_name} Guassian Sharp without Noise", original_image, sharp_gauss)
    calc_simil(f"{noise_name} Guassian Sharp with Noise", noise_image, sharp_gauss)
    calc_mean(f"{noise_name} Guassian Sharp with Noise", noise_image, sharp_gauss)

    # Average sharp
    average_sharp = cv2.filter2D(noise_image, -1, kernel)
    show_image(average_sharp, f"{noise_name} Average Sharp")
    cv2.imwrite(f"{file_name}_average_sharp.png", 255 * average_sharp)
    calc_simil(f"{noise_name} Average Sharp without Noise", original_image, average_sharp)
    calc_mean(f"{noise_name} Average Sharp without Noise", original_image, average_sharp)
    calc_simil(f"{noise_name} Average Sharp with Noise", noise_image, average_sharp)
    calc_mean(f"{noise_name} Average Sharp with Noise", noise_image, average_sharp)

    # Median blur
    median_blur = cv2.medianBlur(noise_image, 5)
    show_image(median_blur, f"{noise_name} Median Blur")
    cv2.imwrite(f"{file_name}_median_blur.png", 255 * average_sharp)
    calc_simil(f"{noise_name} Median Blur without Noise", original_image, median_blur)
    calc_mean(f"{noise_name} Median Blue without Noise", original_image, median_blur)
    calc_simil(f"{noise_name} Median Blur with Noise", noise_image, median_blur)
    calc_mean(f"{noise_name} Median Blur with Noise", noise_image, median_blur)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Salt and pepper
def salt_and_pepper(image, probability):
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


# Poisson
def poisson(image):
    noise_image = np.random.poisson(image).astype(np.uint8)
    return image + noise_image


if __name__ == '__main__':
    # Read the image
    image = cv2.imread('face_value.jpg', 0)
    show_image(image, "Original Image")

    # Add salt and pepper noise
    sp_image = salt_and_pepper(image, 0.10)
    show_image(sp_image, "S&P Image")
    cv2.imwrite('sp_noise.png', 255 * sp_image)

    # A poisson noise
    poisson_image = poisson(image)
    show_image(poisson_image, "Poisson Image")
    cv2.imwrite('poisson_noise.png', 255 * poisson_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Remove the noise from the images
    remove_noise("S&P", "sp", sp_image, image)
    remove_noise("Poisson", "poisson", poisson_image, image)
