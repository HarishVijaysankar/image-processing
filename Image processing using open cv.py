#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('pip install opencv-python numpy matplotlib')


# In[12]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[13]:


def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)


# In[14]:


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[15]:


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


# In[16]:


def crop_border(image, border_thickness):
    """
    Crop the border of the image with the specified thickness from all sides.
    
    Parameters:
        image (numpy.ndarray): The input image.
        border_thickness (int): The thickness of the border to crop from all sides.
    
    Returns:
        numpy.ndarray: The image with the border removed.
    """
    if border_thickness >= min(image.shape[:2]) // 2:
        raise ValueError("Border thickness is too large for the given image dimensions.")
    
    (h, w) = image.shape[:2]
    start_row = border_thickness
    start_col = border_thickness
    end_row = h - border_thickness
    end_col = w - border_thickness
    return image[start_row:end_row, start_col:end_col]


# In[30]:


def blur_image(image, kernel_size=(1, 1)):
    return cv2.GaussianBlur(image, kernel_size, 0)


# In[18]:


def get_image_info(image):
    dimensions = image.shape
    number_of_pixels = image.size
    return dimensions, number_of_pixels


# In[31]:


image_path = "C:/Users/muthu/OneDrive/Desktop/nehru.jpg"

# Load the image
image = load_image(image_path)

# Convert to grayscale
gray_image = convert_to_grayscale(image)

# Rotate the image by 45 degrees
rotated_image = rotate_image(gray_image, 45)

# Crop the image (region from (50, 50) to (150, 150))
cropped_image = crop_image(rotated_image, 10, 10, 150, 150)

# Blur the image
blurred_image = blur_image(cropped_image)

# Get image dimensions and number of pixels
dimensions, number_of_pixels = get_image_info(blurred_image)

# Display the original and processed images
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 3, 2)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')

plt.subplot(2, 3, 3)
plt.title('Rotated Image')
plt.imshow(rotated_image, cmap='gray')

plt.subplot(2, 3, 4)
plt.title('Cropped Image')
plt.imshow(cropped_image, cmap='gray')

plt.subplot(2, 3, 5)
plt.title('Blurred Image')
plt.imshow(blurred_image, cmap='gray')

plt.show()

print(f'Dimensions of the image: {dimensions}')
print(f'Number of pixels: {number_of_pixels}')

