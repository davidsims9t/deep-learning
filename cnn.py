# Convolutional Neural Network

# Part 1 - Building the CNN

# Import libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Init CNN
classifier = Sequential()

# Step 1 - Convolution
# 32 3x3 feature maps
# input_shape=(256, 256, 3) 3d array for color image
# activation function - rectifier function
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
# 2x2 size to pool max values
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Add another Convolutional layer for accuracy improvements
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
# Use a power of 2 for units
classifier.add(Dense(units = 128, activation = 'relu'))

# Use sigmoid because we have a binary outcome
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile the CNN
# Use cross entropy because it's categorical data and has a binary outcome
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fit CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255, # All pixel values are between 0 and 1
        shear_range=0.2, # Random transformations
        zoom_range=0.2,
        horizontal_flip=True)

# All pixel values are between 0 and 1
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set', # Path to training images
        target_size=(64, 64), # Size of images
        batch_size=32, # Random sample size
        class_mode='binary') # Binary or more than 2 categories

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000, # No. of images per training set
        epochs=25,
        validation_data=test_set,
        validation_steps=2000) # Images in test set
