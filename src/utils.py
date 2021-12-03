import cv2
import numpy as np
import random
import torch

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

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

def draw_segmentation_map(image, masks, boxes, labels):
#     results = []
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    all_blur = cv2.blur(image, (50,50), 0)
    for i in range(len(masks)):
        # add unblurred pizzas iteratively to the all_blur image
        all_blur[masks[i]>0] = image[masks[i]>0]
        
        # Changing logic:- Now, each image should have only 1 pizza.
        # Un-blur 1 pizza at a time and keep rest blurred.
        # temp = all_blur.copy()
        # temp[masks[i]>0] = image[masks[i]>0]
        # results.append(temp)
    
    return all_blur

