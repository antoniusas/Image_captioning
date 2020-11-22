import numpy as np
import os
import tensorflow as tf
import keras
import keras.backend as K
import sys
import random
import tqdm
import matplotlib.pyplot as plt
import cv2
import pickle
import tqdm

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import layers
from keras import models
from keras.layers import Embedding
from keras.layers import Input
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

# Loading necessary classes
from Data_processor import Data_processor
from Neural_network_model import Neural_network_model

"""
# Testing on random images found in the 'Test_images' - folder #
Param:
    path - directory path to the folder
    cnn_model - Convolutional neural network model
    num_pic - Amount of pictures to test
"""
def random_images(path, cnn_model,num_pic):
    rnd_img = []
    pred_model = cnn_model
    pred_model = Model(inputs=pred_model.inputs, outputs=pred_model.layers[-2].output)
    rnd_int = random.randint(1, len(os.listdir(path)))
    count = 0
    for filename in os.listdir(path):
        if(count > rnd_int):
            break
        rnd_img.append(filename)
    rnd_img = np.random.permutation(rnd_img)

    img_feat_dict = []
    count = 0
    for images in rnd_img:
        if(count >= num_pic):
            break
        orig_img = image.load_img(os.path.join(path, images), target_size=(224, 224))
        img_data = image.img_to_array(orig_img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        img_features = pred_model.predict(img_data)
        img_feat_dict.append(img_features)
        count += 1
    return rnd_img, img_feat_dict

"""
# Loading tokenizer object #
Param:
    tokenizer_name - Tokenizer filename which includes the tokenizer object
    max_length_name - Name of the file specifying the max length for one of the following 3 language models
"""
def load_tokenizer_obj_and_max_length(tokenizer_name, max_length_name):
    with open(str(tokenizer_name) + '.pickle', 'rb') as fp:
        tokenizer = pickle.load(fp)

    with open(str(max_length_name)+'.txt', "r") as fp:
        max_length = int(fp.readline())
    return tokenizer, max_length

"""
# Find the word based on integer number #
Param:
    integer - any arbitrary number
    tokenizer - tokenizer object
"""
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
"""
# Predict the caption for the requested image. #
Param:
    neural_model - The language model
    tokenizer - Tokenizer object
    photo - a feature vector which describes the image
    max_length - max length of the sequence
"""
def predict_caption(neural_model, tokenizer, photo, max_length):
    in_text = '<startseq> '
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = neural_model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word == 'endseq':
            in_text += ' ' + '<endseq>'
            break
        in_text += ' ' + word
    return in_text

"""
# Display the predicted caption with original image, heatmap of the original image and the predicted caption for the image, including original caption #
Param:
    model - The language model
    cnn_model - The Convolutional neural network model
    img_test - An array consisting of images to test
    img_feat_test - An array consisting of feature-vector for images
    tokenizer - Tokenizer object
    max_length - max length of the sequence
"""
def display_predicted_caption(model,cnn_model,img_test, img_feat_test, tokenizer, max_length):
    test_pictures = 5
    count = 1

    fig = plt.figure(figsize=(30,30))
    image_load = load_img(os.path.join(str('./Test_images'), img_test), target_size=(224, 224,3))
    ax = fig.add_subplot(1,3,count,xticks=[],yticks=[])
    ax.imshow(image_load)
    count += 1
    ax = fig.add_subplot(1,3,count,xticks=[],yticks=[])
    heatmap_ResNet50 = display_heatmap(model=cnn_model, image_name=img_test, path=str('./Test_images'), layer_name_activations="activation_49")
    ax.imshow(heatmap_ResNet50)
    count += 1
    caption = predict_caption(neural_model=model, tokenizer=tokenizer, photo=img_feat_test, max_length=max_length)
    ax = fig.add_subplot(1,3,count)
    plt.axis('off')
    ax.plot()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.text(0,0.7,"PREDICTED CAPTION: {}".format(caption),fontsize=10)
    count += 1
    plt.show()

"""
'HEATMAP - Keras' - Retrieves the activation functions from the CNN-model for a specific layer.
Inspired Code taken from: http://www.hackevolve.com/where-cnn-is-looking-grad-cam/
Credits: Saideep
"""
def display_heatmap(model, image_name, path, layer_name_activations):
    model_testing = model
    img = image.load_img(os.path.join(path, image_name), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    pred = model_testing.predict(x)
    class_idx = np.argmax(pred[0])
    class_output = model_testing.output[:, class_idx]
    last_conv_layer = model_testing.get_layer(layer_name_activations)
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model_testing.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(os.path.join(path, image_name))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed_img

def main(_):
    data_manager = Data_processor()
    neural_network_model = Neural_network_model()

    model = None
    cnn_model = None
    tokenizer = None
    max_length = 0
    while(True):
        print("CHOOSE WHICH MODEL TO USE: ")
        print("::::::::::::::::::::::::::::::::::::::::::")
        print("1. CNN-LSTM")
        print("2. CNN-Bidirectional-LSTM")
        print("3. CNN-TimeDistributed-Bidirectional-LSTM")
        print("4. Exit")
        print("::::::::::::::::::::::::::::::::::::::::::")
        command = input("Specify number: ")
        if(command == str(1)):
            print("...Loading...")
            model = neural_network_model.load_weights(model_name_json="Model_Basic", weights_name="Weight_Basic") # CNN-LSTM
            cnn_model = ResNet50(include_top=True)
            tokenizer, max_length = load_tokenizer_obj_and_max_length("tokenizer_Basic", "max_length_BASIC")
            model.summary()
            print("CNN-LSTM Loaded...")
            break
        elif(command == str(2)):
            print("...Loading...")
            model = neural_network_model.load_weights(model_name_json="Model_Bidirectional", weights_name="Weight_Bidirectional") # CNN-Bidirectional-LSTM
            cnn_model = ResNet50(include_top=True)
            tokenizer, max_length = load_tokenizer_obj_and_max_length("tokenizer_Bidirectional", "max_length_Bidirectional")
            model.summary()
            print("CNN-Bidirectional Loaded...")
            break
        elif(command == str(3)):
            print("...Loading...")
            model = neural_network_model.load_weights(model_name_json="Model_TimeDistributed_Bidirectional", weights_name="weight_TimeDistributed_Bidirectional") # CNN-TimeDistributed-Bidirectional-LSTM
            cnn_model = ResNet50(include_top=True)
            tokenizer, max_length = load_tokenizer_obj_and_max_length("tokenizer_TimeDistributed_Bidirectional", "max_length_TimeDistributed_Bidirectional")
            model.summary()
            print("CNN-TimeDistributed-Bidirectional-LSTM Loaded...")
            break
        elif(command == str("Exit") or command == str(4)):
            sys.exit(0)
            break
        else:
            print("\n::::::Please specify one of the model::::::\n")

    print("\n\n\n")
    while(True):
        print("How many images do you want to test? ('Test_images')-folder")
        command = input("Specify number: ")
        if(int(command) < 0):
            print("...NOT POSSIBLE...")
        elif(int(command) > len(os.listdir(str("./Test_images")))):
            print("...Exceeding the the total limit of folder...")
        else:
            break
    number_of_images = int(command)
    rnd_img, rnd_img_feat = random_images(path=str("./Test_images"), cnn_model=cnn_model, num_pic=number_of_images)

    for i in range(number_of_images):
        print("image: {} of {}".format(i+1, number_of_images))
        display_predicted_caption(model=model, cnn_model=cnn_model, img_test=rnd_img[i], img_feat_test=rnd_img_feat[i], tokenizer=tokenizer, max_length=max_length)

if __name__ == '__main__':
    main(None)
