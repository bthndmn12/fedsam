# Code taken from here: https://github.com/ultralytics/ultralytics/issues/3137#issuecomment-1589107713


import cv2
import numpy as np
from PIL import Image, ImageDraw,ImageFont
import random
from matplotlib import pyplot as plt
import os

image_path= "/kaggle/input/pothole-image-segmentation-dataset/Pothole_Segmentation_YOLOv8/train/images/pic-107-_jpg.rf.f0ad7fbe0407cb85527525e503913079.jpg"
annotation_path= "/kaggle/input/pothole-image-segmentation-dataset/Pothole_Segmentation_YOLOv8/train/labels/pic-107-_jpg.rf.f0ad7fbe0407cb85527525e503913079.txt"

# The Helper functions below assume that the image size is (640,640).Hence resizing the image.
#Open the image
img = Image.open(image_path) 
#Resize the image to 640 by 640
img = img.resize((640, 640))
#if you want then you can save the resized image by img.save('resized_image.jpg')

# <--------- Helper functions starts here ----------------------------------->
def maskVisualize(image,mask):
    fontsize = 18
    f, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].imshow(image)
    ax[1].imshow(mask)

#Define the boundary coordinates as a list of (x, y) tuples
'''
def draw_points_on_image(image, points):
  #resize image to 640*640
  resimg = cv2.resize(image, (640,640))
  #iterate for each mask
  for mask in points:
    #Draw each point on the image
    for point in mask:
        cv2.circle(resimg, tuple(point), 1, (0,0,255), -1,)
  #Display the image
  cv2_imshow(resimg)
'''
#convert the mask from the txt file(annotation_path is path of txt file) to array of points making that mask.
def generate_points(annotation_path=''):
    labels=[] # this will store labels
    #we are assuming that the image is of dimension (640,640). then you have annotated it.
    with open(annotation_path, "r") as file:
        points=[]
        for line in file:
            label,lis=line.split()[0],line.split()[1:]
            labels.append(label)
            lis=list(map(float,lis))
        for i in range(len(lis)):
            lis[i]=int(lis[i]*640)
        newlis=[]
        i=0
        while(i<len(lis)):
          #appendint the coordinates as a tuple (x,y)
            newlis.append((lis[i],lis[i+1]))
            i+=2
        points.append(newlis)
        return labels,points


#the below function convert the boundary coordinates to mask array (it shows mask if you pass 1 at show)
#the mask array is required when we want to augument the mask also using albumentation
def convert_boundary_to_mask_array(labels,points, show=0):
    #Create a new image with the same size as the desired mask
    mask = Image.new("L", (640, 640), 0)
    draw = ImageDraw.Draw(mask)
    for i,boundary_coords in enumerate(points):
      #boundary_coords represent boundary of one polygon
      #Draw the boundary on the mask image
        draw.polygon(boundary_coords,fill=1)
      #Also put the label as text 
      #Compute the centroid of the polygon
        centroid_x = sum(x for x, _ in boundary_coords) / len(boundary_coords)
        centroid_y = sum(y for _, y in boundary_coords) / len(boundary_coords)
        centroid = (int(centroid_x), int(centroid_y))
      #Write the name at the centroid
        text = str(labels[i])
      #Write the label at the centroid
        font = ImageFont.load_default()
        text_w, text_h = draw.textsize(text, font=font)
        text_pos = (centroid[0] - text_w/2, centroid[1] - text_h/2)
        draw.text(text_pos, text, font=font, fill='black')
    #Convert the mask image to a numpy array
    mask_array = np.array(mask)*255
    #Show the mask image
    if(show==1):
      #Image.fromarray(mask_array).show()
        cv2.imshow(mask_array)
    return mask_array

#function that takes mask path (yolov8 seg txt file) and return mask of an image (shape of mask == shape of image)
def generate_mask(annotation_path='',show=0):
    #pass show=1 for showing the generated mask
    #firstly we generate the points (coordinates) from the annotations
    labels,points=generate_points(annotation_path)
    #once we get the points we will now generate the mask image from these points (binary mask image (black/white))
    #mask is represented by white and ground is represented as black
    mask_array=convert_boundary_to_mask_array(labels,points,show)
    return mask_array
# <---------- Helper Functions Ends here ------------------------------------------------------------->

# mask_array=generate_mask(annotation_path=annotation_path,show=0)
# maskVisualize(np.array(img),mask_array)
def process_directory(image_dir, annotation_dir, output_dir):
    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(image_dir, image_file)

        # Construct the corresponding annotation file path
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        annotation_path = os.path.join(annotation_dir, annotation_file)

        # Construct the output file path
        output_file = os.path.splitext(image_file)[0] + '_mask.jpg'
        output_path = os.path.join(output_dir, output_file)

        # Open the image
        img = Image.open(image_path) 
        # Resize the image to 640 by 640
        img = img.resize((640, 640))

        # Generate the mask
        mask_array = generate_mask(annotation_path=annotation_path, show=0)

        # Save the mask to the output directory
        mask_image = Image.fromarray(mask_array)
        mask_image.save(output_path)

# Example usage:
# process_directory('/path/to/images', '/path/to/annotations', '/path/to/output')
