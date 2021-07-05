# importing modules
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Loading model
model1 = load_model('BT_MODEL-v1.h5')

# Preprocessing the Training set
train_data = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_data.flow_from_directory('../Dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
# Preprocessing the Testing set
test_data = ImageDataGenerator(rescale = 1./255)
test_set = test_data.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
# Predicting using the above created model
pred_image = image.load_img('..Dataset/prediction_set/pred2_y.jpg', target_size=(64,64))
pred_image = image.img_to_array(pred_image)
pred_image = np.expand_dims(pred_image, axis = 0)
result = model1.predict(pred_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'yes'
else:
    prediction = 'no'
print(prediction)
import numpy 
import cv2
from scipy.signal import convolve2d # Import convolve2d 
import matplotlib.pyplot as plt # Module To Create a plot 
f = numpy.array([
    [9,4,0],
    [0,-4,0],
    [0,0,0]
])
img =cv2.imread('..Dataset/prediction_set/pred2_y.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cimg = convolve2d(img, f) # Convolve cimbines both n reduces pixels of images
plt.imshow(cimg) # After edges detected # Shows image with help of plot in gray color
