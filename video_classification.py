import numpy as np
from classifiers import MesoInception4
# from tensorflow.keras.models import load_model
from pipeline import compute_accuracy

# Load the trained model
classifier = MesoInception4(learning_rate=0.001)
classifier.load('trained_model.h5')

# Assuming 'video_path' is the path to the video you want to predict
video_path = 'test_videos'

# Make predictions on the video
predictions = compute_accuracy(classifier, video_path)

# Print predictions
for video_name in predictions:
    print('`{}` video class prediction:'.format(video_name), predictions[video_name][0])
