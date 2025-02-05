# Hands-On Project: Building  an Image Filter
# In this project, we will build an image filter that applies a series of image processing techniques to an input image. The filter will perform the following operations:
# Read an image
# Convert the image to grayscale
# Apply edge detection using the Canny algorithm
# combine these operations to created a stylised effect

# import the required libraries
import cv2
import numpy as np

def stylize_image(image_path, output_path):
    # read the image from the disk
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open image.jpg")
        return
    
    # convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Invert the edges to get a sketch effect
    inverted_edges = cv2.bitwise_not(edges) # bitwise_not is used to invert the pixel values of the image. 
    # we need to do this because the Canny edge detector returns white pixels for edges and black pixels for the background, but we want the opposite.
    
    # Combine the original image with the inverted edges (convert to color)
    inverted_edges_color = cv2.cvtColor(inverted_edges, cv2.COLOR_GRAY2BGR) # convert the inverted edges to a color image hence why we are using GRAY2BGR
    stylize_image = cv2.bitwise_and(image, inverted_edges_color) # bitwise_and is used to combine the original image with the inverted edges
    
    # Save and display the stylized image
    cv2.imwrite(output_path, stylize_image)
    cv2.imshow("Stylized Image", stylize_image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    
    
# Run the stylize_image function with the input and output paths
input_image = "image.jpg"
output_image = "stylized_image.jpg"
stylize_image(input_image, output_image)
    
    