#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd


# In[2]:


path_inception = f"{getcwd()}/../tmp2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"


# In[5]:


# Import the inception model  
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = 'tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None
)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
  layer.trainable = False

# Print the model summary
pre_trained_model.summary()


# In[6]:


last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# In[46]:


#defining the data directories
train_data_dir= 'data/train'
validation_data_dir= 'data/validation'
n_training_sample= 400
n_validation_sample= 100
epochs=20
batch_size=10


# In[47]:


from tensorflow.keras.optimizers import RMSprop

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(), 
  tf.keras.layers.Dense(128, activation=tf.nn.relu), 
  tf.keras.layers.Dense(64, activation=tf.nn.relu),
  tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)])

model.compile(
    optimizer=RMSprop(lr=0.0001), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

model.summary()


# In[48]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

# dimensions of our images.
img_width, img_height = 150,150

#defining the data directories
train_dir= 'data/train'
validation_dir= 'data/validation'


# In[49]:


train_datagen = ImageDataGenerator(
    rescale = 1./255.)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(
    rescale = 1./255.
)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=64,
    class_mode='binary',
    target_size=(150,150)
)     

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(
    validation_dir,
    batch_size=64,
    class_mode='binary',
    target_size=(150,150)
)


# In[50]:


model.fit_generator(
    train_generator,
    steps_per_epoch=n_training_sample // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=n_validation_sample // batch_size)


# In[39]:


#testing the model
from keras.preprocessing import image
import numpy as np
pred= image.load_img('data/t2.jpeg', target_size=(150,150))
pred=image.img_to_array(pred)
pred= np.expand_dims(pred, axis=0)
result= model.predict(pred)
print(result)


# In[40]:


if result[0][0]==1:
    answer='Negative'
else:
    answer='Positive'
print(answer)


# In[ ]:




