from tensorflow.keras.preprocessing import image
import numpy as np
import numpy as np
from classifiers import *
from pipeline import *
import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF.h5')
# Assuming you have loaded your trained model into the 'classifier' variable

# Path to the directory containing the test images
test_images_directory = 'test_images/df'

# List of image filenames in the test_images directory
image_filenames = ['as.jpg','df00204.jpg','df01254.jpg','dum.png','img.png','img2.png','qw.jpg']

for filename in image_filenames:
    # Load and preprocess the image
    img_path = os.path.join(test_images_directory, filename)
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Rescale to match the training data preprocessing

    # Get model predictions
    prediction = classifier.predict(img_array)

    # Print the results
    print(f"Image: {filename}")
    print(f"Predicted class: {prediction}")
    print("Real class: Fake", )  # Replace with the actual class label for each image
    print("\n")
