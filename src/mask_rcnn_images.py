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

from PIL import Image
from utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', 
                    help='path to the input data')
parser.add_argument('-d', '--input_dir', required=True, 
                    help='path to the input data')
parser.add_argument('-o', '--out_dir', required=True, 
                    help='path to the output data')
parser.add_argument('-t', '--threshold', default=0.94, type=float,
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
blurred_files = []
not_blurred_files = []

list_multi_pizza_images = [] # append num of multi-pizzas detected

for image_name in image_files:
    
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

            result = draw_segmentation_map(orig_image, masks, boxes, labels)
        
#             if len(results) > 1:
#                 for i, result in enumerate(results):
#                     name = image_name.split(".jpg")[0] + f"_{i}.png"
#                     save_path = os.path.join(args['out_dir'], name)
#                     cv2.imwrite(save_path, result)
#                 list_multi_pizza_images.append(image_name)
#             else:
            image_name = image_name.replace(".jpg", ".png")
            save_path = os.path.join(args['out_dir'], image_name)
#             cv2.imwrite(save_path, results[0])
            cv2.imwrite(save_path, result)
            
                
            blurred_files.append(image_name)

        else:
            not_blurred_files.append(image_name)
            print("Not Blurred: ", image_name)
        
    except Exception as e:
        print(e)
        error_files.append(image_name)

        
with open("blurred_list.txt", "w") as file:
    for x in blurred_files:
        file.write(x + "\n")
        
with open("not_blurred_list.txt", "w") as file:
    for x in not_blurred_files:
        file.write(x + "\n")
        
with open("error_files.txt", "w") as file:
    for x in error_files:
        file.write(x + "\n")

# with open("multi_pizza_images.txt", "w") as file:
#     for x in list_multi_pizza_images:
#         file.write(x + "\n")
        
print("Finished!!")
print("len(blurred_files) = ", {len(blurred_files)})
print("len(not_blurred_list) = ", {len(not_blurred_files)})      
print("len(error_files) = ", {len(error_files)})      
# print("Images with multiple pizzas: ", len(list_multi_pizza_images))
