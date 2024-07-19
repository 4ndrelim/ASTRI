import numpy as np
import os
import cv2

base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, 'illumination.png')
# read input as grayscale
img = cv2.imread(image_path, 0)

# convert image to floats and do dft saving as complex output
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)

# apply shift of origin from upper left corner to center of image
dft_shift = np.fft.fftshift(dft)

# extract magnitude and phase images
mag, phase = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])

# get spectrum for viewing only
spec = np.log(mag) / 30

# NEW CODE HERE: raise mag to some power near 1
# values larger than 1 increase contrast; values smaller than 1 decrease contrast
mag = cv2.pow(mag, 1.1)

# convert magnitude and phase into cartesian real and imaginary components
real, imag = cv2.polarToCart(mag, phase)

# combine cartesian components into one complex image
back = cv2.merge([real, imag])
# print("read", real)
# print("imaginary", imag)
# print("mag", mag)

# shift origin from center to upper left corner
back_ishift = np.fft.ifftshift(back)

# do idft saving as complex output
img_back = cv2.idft(back_ishift)

# combine complex components into original image again
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

# re-normalize to 8-bits
min, max = np.amin(img_back, (0,1)), np.amax(img_back, (0,1))
print(min,max)
img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow("ORIGINAL", img)
cv2.imshow("MAG", mag)
cv2.imshow("PHASE", phase)
cv2.imshow("SPECTRUM", spec)
cv2.imshow("REAL", real)
cv2.imshow("IMAGINARY", imag)
cv2.imshow("COEF ROOT", img_back)
cv2.waitKey(0)
cv2.destroyAllWindows()

# write result to disk
cv2.imwrite(os.path.join(base_dir, "lena_grayscale_opencv.png"), img)
cv2.imwrite(os.path.join(base_dir, "lena_grayscale_coefroot_opencv.png"), img_back)
    