
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

# ! 3. Basic Image Operatiosn with OpenCV

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