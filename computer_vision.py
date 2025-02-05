
# ! 1. What is computer vision? 
# computer vision is the field of AI that enables computers to interpret and process visual information from the world such as images and videos. 
# Key applications include:
# Object detection and recognition
# Image segmentation which is the process of partitioning an image into multiple segments or sets of pixels
# Face recognition
# Image filtering and transformation which is the process of enhancing an image or extracting useful information from it


# ! 2. Installing the Required Libraries
# THe most widely used library for computer vision in python is OpenCV.
# additionally numPy is used for numerical operations
# To install these libraries, run the following commands:
# pip install opencv-python-headless numpy

# Notes: 
# opencv-python-headless is a version of OpenCV without GUI support(useful in server environments or headless environments). headless environments are those that do not have a graphical user interface(GUI) such as a monitor or display attached to them.
# for GUI support to open image windows, we can install opencv-python instead of opencv-python-headless

#! 3. Basic Image Operatiosn with OpenCV

# 3.1 Reading and Displaying an Image 
# going to install GUI support for OpenCV to display images in a window to make code works as expected

# import the required libraries
import cv2

# read an image from the disk
image = cv2.imread("image.jpg")

# check if the image was successfully loaded
if image is None:
    print("Error: Could not open image.jpg")
else: 
    # display the image in a window
    cv2.imshow("Image", image)
    # wait for a key press
    cv2.waitKey(0) # 0 means wait indefinitely for a key press
    # close the window
    cv2.destroyAllWindows()

    # Save the image to a new file
    cv2.imwrite("new_image.jpg", image)
    
    
#! 3.2 Converting an Image to Grayscale
# converting to grayscal is a common preprocessing step in computer vision

# we will use the import and read the image from the previous example

# convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # BGR is the default color space used by OpenCV and it stands for Blue, Green, Red color channels and is a form of RGB color space

# save and display the grayscale image
cv2.imwrite("gray_image.jpg", gray_image)
cv2.imshow("Gray Image", gray_image)
cv2.waitKey(0) # wait indefinitely for a key press
cv2.destroyAllWindows() # close the window

# ! 3.3 Edge Detection using the Canny Algorithm
# Edge detection is a technique used to detect the boundaries of objects within an image. It is a fundamental technique in computer vision and is used in many applications such as object detection, image segmentation, and image filtering.
# The Canny algorithm is a popular edge detection algorithm that was developed by John F. Canny in 1986. It is known for its accuracy and efficiency in detecting edges in images.

# we will again use the import and read the image and convert to grayscale from the previous example

# Apply Guassian blur to the image to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0) # (5, 5) is the kernel size and 0 is the standard deviation of the kernel. kernel means a matrix that is used to perform operations such as blurring, sharpening, edge detection, etc.

# Apply the Canny edge detector
edges = cv2.Canny(blurred, threshold1=50, threshold2=150) # 50 and 150 are the thresholds for the hysteresis procedure used by the Canny edge detector. The hysteresis procedure is used to suppress weak edges and preserve strong edges in the image.

# Save and display the edges
cv2.imwrite("edges.jpg", edges)
cv2.imshow("Edge Detection", edges)
cv2.waitKey(0) # wait indefinitely for a key press
cv2.destroyAllWindows() # close the window


# ! 3.4 Resizing and Rotating an Image
# resizing and rotating images are common operations in computer vision and are used in many applications such as object detection, image classification, and image segmentation.

# we will again use the import and read the image from the previous example

# Resize the image to 50% of its original size
resized = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) # fx and fy are the scaling factors in the x and y directions, respectively. 
# INTER_LINEAR is the interpolation method used to resize the image. none means that the output size is calculated based on the scaling factors.
cv2.imwrite("resized_image.jpg", resized)

# Rotate the image by 45 degrees
(h, w) = image.shape[:2] # get the height and width of the image. shape returns a tuple with the number of rows, columns, and channels of the image and 2 means we are only interested in the first two elements of the tuple
center = (w // 2, h // 2) # calculate the center of the image. // is the floor division operator which returns the integer part of the division and is used to divide the width and height by 2.
M = cv2.getRotationMatrix2D(center, angle=45, scale=1.0) # get the rotation matrix for rotating the image by 45 degrees around the center of the image with a scale factor of 1.0
rotated = cv2.warpAffine(image, M, (w, h)) # apply the rotation matrix to the image using the warpAffine function. 
# warpAffine function is a general function for applying an affine transformation to an image. affine means that the transformation preserves parallel lines and ratios of distances between points.
cv2.imwrite("rotated_image.jpg", rotated) # save the rotated image to a new file
