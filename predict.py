import os
import cv2
import numpy as np 
from os import listdir
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

PATH = r'D:\CNN_leaf\resnet50-final-8.h5'

IMAGE_SIZE = (224,224)

model = load_model(PATH)
print(model.summary())


dict_hot = {'AesculusCalifornica':0,'ErodiumSp':1,'MagnoliaGrandiflora':2}
ss = ['AesculusCalifornica','ErodiumSp','MagnoliaGrandiflora']
inv_map = {v: k for k, v in dict_hot.items()}
label = []
label_hot = []
images = []

for sub in ss:
    for i in listdir(r'D:\CNN_leaf\Dataset\Test\{}'.format(sub)):
        class_name = i[:i.index("_")]
        label.append(class_name)
        label_hot.append(dict_hot[class_name])
    
        img = cv2.imread(os.path.join(r'D:\CNN_leaf\Dataset\Test\{}'.format(sub), i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        images.append(img)
    
testData = np.array(images) / 255.0
label_hot = np.array(label_hot)
testLabel_hot = np_utils.to_categorical(label_hot)


result = model.predict(testData)
print(result)

loss, acc = model.evaluate(testData,testLabel_hot)
print(f"Loss:{loss}")
print(f"Accuracy:{acc}")





