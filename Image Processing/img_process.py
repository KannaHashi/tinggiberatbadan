import cv2
import numpy as np

def image_processing(image):
    h, w = image.shape[:2]
    print("height = {}, width = {}".format(h, w))

    gray = cv2.cvtColor(image, cv2)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5, 5), np.uint8)
    processed_image = cv2.dilate(gray_image, kernel, iterations=1)

    return processed_image

# Fungsi untuk menghitung tinggi badan
def calculate_height(image):
    height = 170  # Contoh nilai tinggi badan dalam satuan cm

    return height

def calculate_weight(image):
    weight = 65  # Contoh nilai berat badan dalam satuan kg

    return weight

def display_result(height, weight):
    # Menampilkan hasil perhitungan tinggi dan berat badan
    print("Tinggi badan: ", height, " cm")
    print("Berat badan: ", weight, " kg")

# Main program
if __name__ == "__main__":
    # Load gambar
    image = cv2.imread('../img/1-1.jpg')

    # Proses image processing dan morphological image processing
    processed_image = image_processing(image)

    # Hitung tinggi badan
    height = calculate_height(processed_image)

    # Hitung berat badan
    weight = calculate_weight(processed_image)

    # Menampilkan hasil perhitungan
    display_result(height, weight)
