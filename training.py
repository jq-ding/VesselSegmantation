###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################
import cv2
from keras import backend as K
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import configparser
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, core, Dropout
from keras.callbacks import ModelCheckpoint

import sys
sys.path.insert(0, './lib/')
from help_functions import *

# function to obtain data for training/testing (validation)
from extract_patches import get_data_training
from keras.utils import plot_model
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

#Define the neural network
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2, patch_height*patch_width))(conv6)
    conv6 = core.Permute((2, 1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

#//conv7=conv7+np

    model = Model(inputs=inputs, outputs=conv7)

    # # weighted loss
    # def weighted_concat(y_true, matrix):
    #     y_true = tf.concat([y_true, matrix], 1)
    #     return y_true
    # def weighted_loss(y_true, y_pred):
    #     weight = y_true[:, 3, :, :]
    #     y_true = y_true[:, 0:3, :, :]
    #     epsilon = K.epsilon()
    #     y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    #     loss = -y_true * K.log(y_pred) * weight
    #     loss = K.sum(loss, axis=1)
    #     return loss
    #
    #     return weighted_loss


    # focal loss
    def catergorical_focal_loss(gamma=2.0, alpha=0.25):
        """
        Implementation of Focal Loss from the paper in multiclass classification
        Formula:
            loss = -alpha*((1-p_t)^gamma)*log(p_t)
        Parameters:
            alpha -- the same as wighting factor in balanced cross entropy
            gamma -- focusing parameter for modulating factor (1-p)
        Default value:
            gamma -- 2.0 as mentioned in the paper
            alpha -- 0.25 as mentioned in the paper
        """

        def focal_loss(y_true, y_pred):
            # Define epsilon so that the backpropagation will no result in NaN
            # for o divisor case

            epsilon = K.epsilon()
            # Add the epsilon to prediction value
            # y_pred = y_pred + epsilon
            # Clip the prediction value
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
            # Calculate cross entropy
            cross_entropy = -y_true * K.log(y_pred)

            # Calculate weight that consists of modulating factor and weighting factor
            weight = alpha * y_true * K.pow((1 - y_pred), gamma)
            # Calculate focal loss
            loss = weight * cross_entropy
            # Sum the losses in mini_batch
            loss = K.sum(loss, axis=1)
            return loss

        return focal_loss

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model




#========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration.txt')
#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))



#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original=path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth=path_data + config.get('data paths', 'train_groundTruth'),  #masks
    patch_height=int(config.get('data attributes', 'patch_height')),
    patch_width=int(config.get('data attributes', 'patch_width')),
    N_subimgs=int(config.get('training settings', 'N_subimgs')),
    inside_FOV=config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)


#========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0], 40)
# visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5), '/'+name_experiment+'/'+"sample_input_imgs")#.show()
# visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5), '/'+name_experiment+'/'+"sample_input_masks")#.show()


#=========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
model = get_unet(n_ch, patch_height, patch_width)  #the U-net model

plot_model(model, to_file='model_auth.png', show_shapes=True)
plt.savefig('C:/Users/DJQ/Downloads/Retina-Unet-master (1)/Retina-Unet-master/model_auth.png' )


print("Check: final output of the network:")
print(model.output_shape)
# plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment + '_architecture.json', 'w').write(json_string)



#============  Training ==================================
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment + '_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased


# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

patches_masks_train = masks_Unet(patches_masks_train)  # reduce memory consumption
model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer])


#========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
#test the model
# score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
