'''This is a simple but powerful image classification model, hope this will benefit beginners.

In our setup, we will:
1. Data pre-processing:
- Download the data at: 
  https://www.kaggle.com/c/dogs-vs-cats/data
- Create cats-vs-dogs/ folder
- Create training/ and validation/ sub-folders inside cats-vs-dogs/ folder
- Divide the data into cats-vs-dogs/training/ and cats-vs-dogs/validation/ sub-folders
- Create cats/ and dogs/ sub-folders inside cats-vs-dogs/training/ and cats-vs-dogs/validation/, respectively
- Put 11250 cat pictures into training/cats/, 1250 into validation/cats, and put 11250 dog pictures into training/dogs/, 1250 into validation/dogs/
So that we have 11250 training examles for each class, and 1250 validation examples for each class.
In summary, the data structure will be like:
```
cats-vs-dogs/
    training/
        cats/
            cat000.jpg
            cat001.jpg
            ...
        dogs/
            dog000.jpg
            dog001.jpg
            ...
    validation/
        cats/
            cat000.jpg
            cat001.jpg
            ...
        dogs/
            dog000.jpg
            dog001.jpg
            ...
2. Create simple convolutional neural network for image classification
- training and validation process
3. Test with your own samples
```
'''

import os
from shutil import copyfile
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# make sure the code being run and the data downloaded are in the same directory
image_names = os.listdir('train')
# length of data 
print(len(image_names))

# create new sub-folders for storing cat and dog pictures, respectively
os.mkdir('train/cats')
os.mkdir('train/dogs')
for fnames in image_name:
    if fnames.startswith('cat'):
        this_file = 'train/' + fnames
        destination = 'train/cats/' + fnames
        copyfile(this_file,destination)
    else:
        this_file = 'train/' + fnames
        destination = 'train/dogs/' + fnames
        copyfile(this_file,destination)

# create sub-folders
os.mkdir('cats-vs-dogs')
os.mkdir('cats-vs-dogs/training')
os.mkdir('cats-vs-dogs/training/cats')
os.mkdir('cats-vs-dogs/training/dogs')
os.mkdir('cats-vs-dogs/validation')
os.mkdir('cats-vs-dogs/validation/cats')
os.mkdir('cats-vs-dogs/validation/dogs')

# a function for spiltting data 
def Split_Data(DATA,TRAINING,VALIDATION,SPLIT_SIZE):
    files = []
    for fname in os.listdir(DATA):
        files.append(fname)
    shuffled_set = random.sample(files,len(files))
    training_length = int(len(files)*SPLIT_SIZE)
    validation_length = int(len(files)-training_length)
    training_set = shuffled_set[0:training_length]
    validation_set = shuffled_set[-validation_length:]
    for fname in training_set:
        this_file = DATA + fname
        destination = TRAINING + fname
        copyfile(this_file,destination)
    for fname in validation_set:
        this_file = DATA + fname
        destination = VALIDATION + fname
        copyfile(this_file,destination)  
        
CATS_DATA = 'train/cats/'
CATS_TRAINING = 'cats-vs-dogs/training/cats/'
CATS_VALIDATION = 'cats-vs-dogs/validation/cats/'
DOGS_DATA = 'train/dogs/'
DOGS_TRAINING = 'cats-vs-dogs/training/dogs/'
DOGS_VALIDATION = 'cats-vs-dogs/validation/dogs/'    
  
Split_Data(CATS_DATA,CATS_TRAINING,CATS_VALIDATION,0.9)
Split_Data(DOGS_DATA,DOGS_TRAINING,DOGS_VALIDATION,0.9)

# check the number of examples in each sub-folder
training_cats_names = os.listdir(CATS_TRAINING)
training_dogs_names = os.listdir(DOGS_TRAINING)
validation_cats_names = os.listdir(CATS_VALIDATION)
validation_dogs_names = os.listdir(DOGS_VALIDATION)
print(len(training_cats_names))
print(len(training_dogs_names))
print(len(validation_cats_names))
print(len(validation_dogs_names))

# pre-processing to images data
train_dir = 'cats-vs-dogs/training'
# use ImageDataGenerator() to rescale, of course you can also do data augmentation
train_datagen = ImageDataGenerator(rescale=1/255)
# use flow_from_directory() to generate batches of image data and their labels from images in their respective folders
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=250,class_mode='binary')
validation_dir = 'cats-vs-dogs/validation'
validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=25,class_mode='binary')

# model contains three convolutional layers, and two fully-connected layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
# use 'adam' as the optimizer, 'binary_crossentropy' as the loss function to train model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# start training!
model.fit(train_generator,steps_per_epoch=22500/250,epochs=50,verbose=1,validation_data=validation_generator,validation_steps=2500/25)
# save generated weights
model.save_weights('weights.h5')

# test the trained model with your own samples
test_dir = 'test_cats_and_dogs'
test_names = os.listdir(test_dir)
img_ori = os.path.join(test_dir,test_names[0])
img = image.load_img(img_ori,target_size=(150,150))
img_data = image.img_to_array(img)
img_data = img_data/255
plt.imshow(img_data)
pic = np.expand_dims(img_data,axis=0)
output = model.predict(pic)
print(output)
if output>0.5:
    print('dog')
else:
    print('cat')
