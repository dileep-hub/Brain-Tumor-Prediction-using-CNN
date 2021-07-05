# importing modules
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Loading model
model1 = load_model('BT_MODEL.h5')

# Preprocessing the Training set
train_data = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
# Preprocessing the Testing set
test_data = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
# Predicting using the above created model
pred_image = image.load_img('dataset/kag2y.jpg', target_size=(64,64))
pred_image = image.img_to_array(pred_image)
pred_image = np.expand_dims(pred_image, axis = 0)
result = model1.predict(pred_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'yes'
else:
    prediction = 'no'
print(prediction)