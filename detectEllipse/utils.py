import cv2
import numpy as np
import random
import torch

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
from h_ellipse import hough_e
from detect_ellipse import detect_ellipse

# this will help us create a different color for each class
# COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)
    
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    
    # get the classes labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    
    only_pizza_scores = []
    only_pizza_labels = []
    only_pizza_masks = []
    only_pizza_boxes = []
    
    for idx, label in enumerate(labels):
        if label == 'pizza':
            only_pizza_scores.append(scores[idx])
            only_pizza_labels.append(labels[idx])
            only_pizza_masks.append(masks[idx])
            only_pizza_boxes.append(boxes[idx])

    only_pizza_masks = np.array(only_pizza_masks)

    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [i for i in only_pizza_scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    
    # discard masks for objects which are below threshold
    only_pizza_masks = only_pizza_masks[:thresholded_preds_count]
    # discard bounding boxes below threshold value
    only_pizza_boxes = only_pizza_boxes[:thresholded_preds_count]
    only_pizza_labels = only_pizza_labels[:thresholded_preds_count]
    only_pizza_scores = only_pizza_scores[:thresholded_preds_count]
    
    return only_pizza_masks, only_pizza_boxes, only_pizza_labels, only_pizza_scores


def mark_circles(image):
    params = cv2.SimpleBlobDetector_Params()
    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 10

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.4

    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.02

    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.001
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(image)
    
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)
    text = "Number of Circular Blobs: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    
    return blobs

def hough_circles(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_circles = cv2.HoughCircles(img, 
    cv2.HOUGH_GRADIENT, 1, 120, param1 = 100,
    param2 = 30, minRadius = 0, maxRadius = 0)
  
    # Draw circles that are detected.
    if detected_circles is not None:
        print("circle detected")

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(image, (a, b), r, (0, 255, 0), 10)

#             # Draw a small circle (of radius 1) to show the center.
#             cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
#             cv2.imshow("Detected Circle", img)
#             cv2.waitKey(0)
    return image

def hough_circles2(image):
    print("hough_circles 2!")
    img = image
    # Read image as gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise
    img_blur = cv2.medianBlur(gray, 15)
    # Apply hough transform on the image
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=10, minRadius=100, maxRadius=3000)
    # Draw detected circles
    if circles is not None:
        print("circles detected")
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw inner circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
            
    return img

def detect_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print("No. of contours: ", len(contours))
    i = 0

    # list for storing names of shapes
    for contour in contours:

#         # here we are ignoring first counter because 
#         # findcontour function detects whole image as shape
#         if i == 0:
#             i = 1
#             continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), False)

        print("Len of approx poly: ", len(approx))
        
        # using drawContours() function
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])

        # putting shape name at center of each shape
        if len(approx) == 3:
            cv2.putText(img, 'Triangle '+str(len(approx)), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        elif len(approx) == 4:
            cv2.putText(img, 'Quadrilateral '+str(len(approx)), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        elif len(approx) == 5:
            cv2.putText(img, 'Pentagon '+str(len(approx)), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        elif len(approx) == 6:
            cv2.putText(img, 'Hexagon '+str(len(approx)), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        else:
            cv2.putText(img, 'circle '+str(len(approx)), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return img

def detect_circles(image, masks, boxes, labels):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(image.shape)
    
    blank_image = np.zeros(image.shape, np.uint8)
    
    white_image = np.ones(image.shape, np.uint8)*255
    
    for i in range(len(masks)):
        blank_image[masks[i]>0] = white_image[masks[i]>0]
    
    is_good, detected_ellipse = detect_ellipse(blank_image)
    print("Keep it? ", is_good)
    
    if is_good:
        #cropped_pizza = np.zeros(image.shape, np.uint8) #background will be black
        cropped_pizza = cv2.blur(image, (50, 50), 0) # blurred-background
        for i in range(len(masks)):
            cropped_pizza[masks[i]>0] = image[masks[i]>0]
        return True, cropped_pizza
    
    return False, blank_image

def draw_segmentation_map(image, masks, boxes, labels):
    results = []
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    all_blur_individual = cv2.blur(image, (50,50), 0)
    all_blur_combined = cv2.blur(image, (50, 50), 0)
    
    for i in range(len(masks)):
        # Un-blur 1 pizza at a time and keep rest blurred.
        temp = all_blur_individual.copy()
        temp[masks[i]>0] = image[masks[i]>0]
        results.append(temp)
        
        # add unblurred pizzas iteratively to a combined output image
        all_blur_combined[masks[i]>0] = image[masks[i]>0]
        
    if(len(masks) > 1):
        # only append if there were multiple pizzas. Otherwise it will be duplicate
        results.append(all_blur_combined)
    
    return results

