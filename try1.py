import numpy as np
from classifiers import MesoInception4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pipeline import compute_accuracy

class CustomImageDataGenerator(ImageDataGenerator):
    def __init__(self, frame_skip_prob=0.2, add_random_frames_prob=0.2, frame_rate_change_prob=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_skip_prob = frame_skip_prob
        self.add_random_frames_prob = add_random_frames_prob
        self.frame_rate_change_prob = frame_rate_change_prob

    def random_transform(self, x, seed=None):
        x = super().random_transform(x, seed)

        # Frame reversal with a 50% chance
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=2)  # Assuming channel is the last axis (axis=2)

        # Frame skipping
        if np.random.rand() < self.frame_skip_prob:
            x = np.delete(x, np.arange(0, x.shape[0], 2), axis=0)

        # Adding random frames
        if np.random.rand() < self.add_random_frames_prob:
            num_random_frames = np.random.randint(1, 5)
            random_frames = np.random.uniform(0, 1, size=(num_random_frames,) + x.shape[1:])
            x = np.concatenate([x, random_frames], axis=0)

        # Frame rate change
        if np.random.rand() < self.frame_rate_change_prob:
            frame_rate_change_factor = np.random.uniform(0.5, 2.0)
            num_frames = int(x.shape[0] * frame_rate_change_factor)
            x = np.resize(x, (num_frames,) + x.shape[1:])

        return x

# 1 - Load the model and its pretrained weights
classifier = MesoInception4(learning_rate=0.001)
classifier.load('trained_model.h5')

# 2 - Minimal image generator
batch_size = 32
target_size = (256, 256)

# Use the custom data generator with additional augmentations
dataGenerator = CustomImageDataGenerator(
    rescale=1./255,
    frame_skip_prob=0.2,
    add_random_frames_prob=0.2,
    frame_rate_change_prob=0.2
)

combined_generator = dataGenerator.flow_from_directory(
    'deepfake_database/train_test',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = dataGenerator.flow_from_directory(
    'deepfake_database/validation',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Step 4: Model Training
epochs = 500 # Adjust as needed

# Assuming your Classifier class has been updated to accept validation_data
classifier.fit(
    combined_generator,
    epochs=epochs,
    validation_data=validation_generator
)
classifier.model.save('trained_model1.h5')

# Evaluate on Validation Set
validation_loss, validation_accuracy = classifier.evaluate(validation_generator)
print(f'Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}')

# Uncomment this if you want to make predictions on test videos
# predictions = compute_accuracy(classifier, 'test_videos')
# for video_name in predictions:
#     print('`{}` video class prediction:'.format(video_name), predictions[video_name][0])
