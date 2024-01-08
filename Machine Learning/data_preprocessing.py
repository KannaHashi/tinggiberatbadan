import cv2

img = cv2.imread('dataset_img/1-3.jpg')

grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale Image', grayscale_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('dataset_img/1-3_grayscale.jpg', grayscale_image)
