'''
Notes:
input_dir /common/home/dv347/Data/Pizza10/images
out_dir /common/home/dv347/Data/Pizza10/blurred_output
reference: https://debuggercafe.com/instance-segmentation-with-pytorch-and-mask-r-cnn/
'''


import torch
import torchvision
import cv2
import argparse, os
from os import listdir
from os.path import isfile, join
import numpy as np

from PIL import Image
from utils import draw_segmentation_map, get_outputs, detect_circles
from torchvision.transforms import transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--input_dir', default="/common/home/dv347/Data/Pizza10/images", 
                    help='path to the input data')
parser.add_argument('-o', '--out_dir', required=True, 
                    help='path to the output data')
parser.add_argument('-t', '--threshold', default=0.90, type=float,
                    help='score threshold for discarding detection')
args = vars(parser.parse_args())

# initialize the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, 
                                                           num_classes=91)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()

# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])


img_dir = args['input_dir']
image_files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]

error_files = []
no_pizza_segmented = []
elliptical = []
not_elliptical = [] # append num of multi-pizzas detected

dir_elliptical_pizzas = os.path.join(args['out_dir'], "elliptical_pizzas")
dir_not_elliptical_pizzas = os.path.join(args['out_dir'], "not_elliptical_pizzas")


isExist = os.path.exists(args['out_dir'])
if not isExist:
    os.makedirs(args['out_dir'])
    os.makedirs(dir_elliptical_pizzas)
    os.makedirs(dir_not_elliptical_pizzas)
    print("The new directory is created!")
else:
    print("Output Directory already exists. Exiting...")
    exit()
    

for image_name in image_files:
    print("\n processing img: ", image_name)
    try:
        image_path = os.path.join(img_dir, image_name)
        
        image = Image.open(image_path).convert('RGB')

        orig_image = image.copy()

        # transform the image
        image = transform(image)
        # add a batch dimension
        image = image.unsqueeze(0).to(device)

        masks, boxes, labels, scores = get_outputs(image, model, args['threshold'])
        
        assert len(masks) == len(scores)
        if len(scores) > 0:
            
            is_good, output = detect_circles(orig_image, masks, boxes, labels)
                 
            if is_good:
                name = image_name.replace(".jpg", ".png")
                save_path = os.path.join(dir_elliptical_pizzas, name)
                cv2.imwrite(save_path, output)
                elliptical.append(image_name)
            
            else:
                # save original image in case ellipse not detected.
                orig_image = np.array(orig_image)
                orig_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(dir_not_elliptical_pizzas, image_name)
                cv2.imwrite(save_path, orig_image)
                not_elliptical.append(image_name)

        else:
            no_pizza_segmented.append(image_name)
            print("No pizza segmented: ", image_name)
        
    except Exception as e:
        print(e)
        error_files.append(image_name + f" {e}")

        
with open(os.path.join(args['out_dir'], "elliptical_list.txt"), "w") as file:
    for x in elliptical:
        file.write(x + "\n")
        
with open(os.path.join(args['out_dir'], "not_elliptical_list.txt"), "w") as file:
    for x in not_elliptical:
        file.write(x + "\n")
        
with open(os.path.join(args['out_dir'], "error_files.txt"), "w") as file:
    for x in error_files:
        file.write(x + "\n")

with open(os.path.join(args['out_dir'], "no_pizza_segmented_list.txt"), "w") as file:
    for x in no_pizza_segmented:
        file.write(x + "\n")
        
print("Finished!!")
print("len(elliptical) = ", {len(elliptical)})
print("len(not_elliptical) = ", {len(not_elliptical)})      
print("len(error_files) = ", {len(error_files)})      
print("len(no_pizza_segmented) = ", len(no_pizza_segmented))
