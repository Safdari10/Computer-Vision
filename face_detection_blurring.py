# HandsOn-OpenCV: Face Detection and Blurring

# In this hands-on project, you will learn how to detect faces in images and blur them using OpenCV. You will use the Haar Cascade classifier to detect faces and apply a Gaussian blur to the detected faces.

# Need to install OpenCV and numpy
# pip install opencv-python-headless numpy but we want to use the GUI so we will install opencv-python instead



import cv2
import numpy as np

def detect_and_blur_faces(image_path, output_path, cascade_path="haarcascade_frontalface_default.xml"):
    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open image.jpg")
        return
    
    # Convert the image to grayscale (Haar Cascade works on grayscale images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 
    # detectMultiScale is used to detect objects in an image. It returns a list of rectangles where objects were detected.
    # scaleFactor: Parameter specifying how much the image size is reduced at each image scale. It is used to create a scale pyramid. Default value is 1.1
    # minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it. This parameter will affect the quality of the detected faces. Higher value results in fewer detections but with higher quality. Default value is 3.
    # minSize: Minimum possible object size. Objects smaller than this size are ignored. This parameter is used to filter out small objects that are not faces. Default value is (30, 30).
    
    print(f"Detected {len(faces)} face(s).")
    
    # Loop over detected faces and apply a blur to each face region
    for (x, y, w, h) in faces:
        # Extract teh face region
        face_region = image[y:y+h, x:x+w]
        
        # Apply a Gaussian blur to the face region
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30) # (99, 99) is the kernel size and 30 is the standard deviation of the kernel
        
        # Replace the original face region with the blurred face
        image[y:y+h, x:x+w] = blurred_face
        
    # Save the resulting image
    cv2.imwrite(output_path, image)
    
    # Optionally, display the image
    cv2.imshow("Blurred Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Example usage:
input_image = "face.jpg"
output_image = "blurred_faces.jpg"
detect_and_blur_faces(input_image, output_image)