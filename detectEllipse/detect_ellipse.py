import matplotlib.pyplot as plt
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import cv2
import numpy as np


def detect_ellipse(image):
    (H, W, _) = image.shape
    
    # Downscale image. Preserve aspect ratio
    w = 128
    scale = int((w*100)/W)
    h = int((H*scale)/100)
    image = cv2.resize(image, (w,h), interpolation=cv2.INTER_AREA)
    
    # Pad boundaries
    pad = 6 #pixels
    image = cv2.copyMakeBorder(image, pad,pad,pad,pad, cv2.BORDER_CONSTANT, (0,0,0))
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #image.rgb2gray(image)

    edges = canny(image_gray, sigma=2.0,low_threshold=0.55, high_threshold=0.8)
    result = hough_ellipse(edges, threshold=150, accuracy=25, min_size=10)

#     result = np.array([x for x in result if x[3] > 10 and x[4]>10]) # Remove flat ellipses

    result.sort(order='accumulator')
    
    if len(result) > 0:
        print("Len of result: ", len(result))
        # Estimated parameters for the ellipse
        best = list(result[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]

        # Draw the ellipse on the original image
#         cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
#         image[cy, cx] = (0, 0, 255)
        
        ellipse_ar = np.pi * (a) * (b)
        pizza_ar = np.sum(image_gray > 2)
        diff_ar = abs(ellipse_ar - pizza_ar)
        print(ellipse_ar)
        print(pizza_ar)
        print(diff_ar)
        
        if diff_ar < 0.15 * pizza_ar:
            
            return True, image
    
    return False, image
    