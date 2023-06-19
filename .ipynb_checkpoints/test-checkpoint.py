import tensorflow as tf
import os #navigate file structures
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model


os.path.join('data', 'happy')

gpus = tf.config.experimental.list_physical_devices('GPU')#'CPU'

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)#avoids Added Memory Error(OOM)
    
#len(gpus) gives number of devices available

#clean data
data_dir = 'data'
os.listdir(data_dir)
image_exts = ['jpeg', 'jpg', 'bmp', 'png', 'svg']

for image_class in os.listdir(data_dir): #loops through folder
    for image in os.listdir(os.path.join(data_dir, image_class)): #loops through subfolders
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path) #removes invalid data from the dataset
        except Exception as e: 
            print('Issue with image {}'.format(image_path))#throws exception 

#Load

data = tf.keras.utils.image_dataset_from_directory('data')#tf.Dataset(API)
#keres-> img preprocessing. 
data_iterator = data.as_numpy_iterator()#Img rep as numpy arr
batch = data_iterator.next() #Batches into Labels and Images

#scale
data = data.map(lambda x,y: (x/255, y))# x-data, y-lavle
data.as_numpy_iterator().next()
#split
len(data)
#batch Sizes
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)
train_size

#split fr
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#Deep Learning- Brains
#Create CNN Model using Sequentisl API
model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))#Convulation, Traverse image data
model.add(MaxPooling2D())#Pooling, Reduce image data

model.add(Conv2D(32, (3,3), 1, activation='relu'))#Convulation
model.add(MaxPooling2D())#Pooling

model.add(Conv2D(16, (3,3), 1, activation='relu'))#Convulation
model.add(MaxPooling2D())#Pooling

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))#0-Happy, 1-Sad

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

#Taking Logs
logdir='logs'#Dir called Logs
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)#Logs can be viewed on TensorBoard
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

#Plot Perf. from logged data
#Loss
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()
#Accuracy
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#Evaluation
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(pre.result(), re.result(), acc.result())

#Testing
img = cv2.imread('any.jpg')#Any img not from the dataset.. Must be saved in the Project Folder.
plt.imshow(img)
plt.show()

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

#Testing2
yhat = model.predict(np.expand_dims(resize/255, 0))
if yhat > 0.5: 
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')

#Savinf
model.save(os.path.join('models','imageclassifier.h5'))
new_model = load_model('imageclassifier.h5')
new_model.predict(np.expand_dims(resize/255, 0))



