import os
import cv2
import numpy as np
import pandas as pd 
from os import listdir
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D ,MaxPooling2D,Dropout
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.utils import np_utils
from keras.models import Model
from keras import optimizers
print("======================================================================")

# 資料路徑
PATH = r'D:\CNN_leaf\Dataset'

# 圖片大小
IMAGE_SIZE = (224,224)

# 辨識類別數
NUM_CLASSES = 3

# 凍結層數
FREEZE_LAYERS = 200

# EPOCH次數
EPOCH = 5

# 批次量
BATCH_SIZE = 32

# 學習率
LEARNING_RATE = 0.001

# 儲存路徑
WEIGHTS_SAVING = 'resnet50-final-8.h5'



dict_hot = {'AesculusCalifornica':0,'ErodiumSp':1,'MagnoliaGrandiflora':2}
label = []
label_hot = []
images = []

for i in listdir(os.path.join(PATH,'Train')):
    # 讀取圖片類別
    class_name = i[:i.index("_")]
    label.append(class_name)
    label_hot.append(dict_hot[class_name])
    
    # 讀取圖片&resize
    img = cv2.imread(os.path.join(PATH,'Train',i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    images.append(np.array(img))
print("Loading Complete\n")
    
images = np.array(images) 
label_hot = np.array(label_hot)

print("images.shape={} , label_hot.shape=={}\n".format(images.shape, label_hot.shape))

# Training Data 80%
(trainData, valiData, trainLabels, valiLabels) = train_test_split(images, label_hot, test_size=0.2, random_state=42)

print("trainData records: {}".format(len(trainData)))
print("valiData records: {}".format(len(valiData)))
print("trainData.shape={} trainLabels.shape={}".format(trainData.shape, trainLabels.shape))
print("valiData.shape={} valiLabels.shape={}\n".format(valiData.shape, valiLabels.shape))

# One hot encoding
trainLabels_hot = np_utils.to_categorical(trainLabels)
valiLabels_hot = np_utils.to_categorical(valiLabels)
print("One hot encoding matrix shape (Train):{}".format(trainLabels_hot.shape))
print("One hot encoding matrix shape (Val):{}\n".format(valiLabels_hot.shape))


datagen = ImageDataGenerator(
    rescale=1.0/255.0,)

#featurewise_center=True,
#featurewise_std_normalization=True,

datagen.fit(trainData)
train_generator = datagen.flow(trainData, trainLabels_hot, shuffle=True, batch_size = BATCH_SIZE)
val_generator = datagen.flow(valiData, valiLabels_hot, shuffle=True, batch_size = BATCH_SIZE)


# 獲得Pre-trained model(使用ResNet50)
Pretrained_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))

# 最後一層FC層攤平
x = Pretrained_model.layers[-1].output
x = Flatten()(x)

# 凍結ResNet
Model_final = Model(inputs=Pretrained_model.input, outputs=x)
for layer in Model_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in Model_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

model = Sequential()
model.add(Model_final)
model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

#categorical_crossentropy
model.compile(optimizer = optimizers.Adam(lr = LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# 一次迭代的批次數目 (on-fly data generator: _倍的資料量)
STEPS_PER_EPOCH = int( (len(trainData) * 1000) / BATCH_SIZE)
VALIDATION_STEPS = int( (len(trainData)* 320)  / BATCH_SIZE)

train_history = model.fit_generator(train_generator,
                        steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCH,
                        validation_data = val_generator, validation_steps = VALIDATION_STEPS)

model.save(WEIGHTS_SAVING)

epo_list = [i for i in range(0,EPOCH)]

plt.figure(1)
plt.title("Loss")
plt.plot(epo_list,train_history.history['loss'],label = 'loss')
plt.plot(epo_list,train_history.history['val_loss'],label = 'val loss')

plt.legend()
plt.xlabel('epochs')
#plt.savefig('train'+str(epc)+'.png')

plt.figure(2)
plt.title("Accuracy")
plt.plot(epo_list,train_history.history['acc'],label = 'accuracy')
plt.plot(epo_list,train_history.history['val_acc'],label = 'val accuracy')
plt.legend()
plt.xlabel('epochs')
#plt.savefig('test'+str(epc)+'.png')
plt.show()