import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import cv2
import pickle

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
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Importing necessary classes
from Data_processor import Data_processor
from Neural_network_model import Neural_network_model

def main(_):
    data_manager = Data_processor()
    neural_network_model = Neural_network_model()

    # Hyper-parameters #
    json_data, unique_id, img_id, img_caption = data_manager.COCO_captions('captions_val2017.json')
    number_of_images = 10
    img_height, img_width = 224, 224
    perc_train = 90 # Percent
    epochs = 5
    batch_size = 8

    coco_img = data_manager.COCO_images(num_img=number_of_images)
    coco_img_seg = data_manager.COCO_images_segmentation(num_img=number_of_images)
    coco_img_cap = data_manager.COCO_caption_ordered(img_array=coco_img, json_data=json_data, num_img=number_of_images)

    coco_img_train, coco_img_test = data_manager.COCO_train_test_set(image_array=coco_img, train_percentage=perc_train, num_img=number_of_images)
    coco_img_seg_train, coco_img_seg_test = data_manager.COCO_train_test_set(image_array=coco_img_seg, train_percentage=perc_train, num_img=number_of_images)

    coco_img_cap_train = data_manager.COCO_caption_ordered(img_array=coco_img_train, json_data=json_data, num_img=number_of_images)
    coco_img_cap_test = data_manager.COCO_caption_ordered(img_array=coco_img_test, json_data=json_data, num_img=number_of_images)
    coco_img_cap_train = data_manager.COCO_img_cap_clean(img_cap_array=coco_img_cap_train, num_img=number_of_images)

    coco_img_cap_start_end_seq = data_manager.start_end_seq_token(img_cap_array=coco_img_cap_train)
    all_words, dictionary, vocabulary_size = data_manager.text_processing(img_cap_array=coco_img_cap_start_end_seq)
    coco_img_cap_tokenize, tokenizer, max_length = data_manager.tokenize_char_to_int(img_cap_array=coco_img_cap_start_end_seq, vocabulary_size=vocabulary_size)

    #cnn_model = VGG16(weights='imagenet', include_top=True)
    #cnn_model = VGG19(weights='imagenet', include_top=True)
    cnn_model = ResNet50(weights='imagenet', include_top=True)
    img_features_train = data_manager.img_feat(image_array=coco_img_train, height=img_height, width=img_width, path=str('./Dataset_COCO_5k_images'), CNN_model=cnn_model)
    img_features_test = data_manager.img_feat(image_array=coco_img_test, height=img_height, width=img_width, path=str('./Dataset_COCO_5k_images'), CNN_model=cnn_model)

    Xtext, Ximage, ytext = data_manager.preprocessing(img_cap_array_tokenized=coco_img_cap_tokenize, img_feature_array=img_features_train, vocab_size=vocabulary_size, max_len=max_length)

    train_percent = int(((len(Xtext)/100)*perc_train))
    test_percent = int((1-train_percent))

    Xtext_train = Xtext[:train_percent,:]
    Ximage_train = Ximage[:train_percent,:]
    ytext_train = ytext[:train_percent,:]

    Xtext_val = Xtext[train_percent:len(Xtext),:]
    Ximage_val = Ximage[train_percent:len(Xtext),:]
    ytext_val = ytext[train_percent:len(Xtext),:]
    print(Ximage_val.shape)
    print(Ximage_train.shape)

    Ximage_train = Ximage_train[:,0,:]
    Ximage_val = Ximage_val[:,0,:]

    model = None
    command = input("Which model to train: ")
    if (int(command) == 1):
        model = neural_network_model.basic_image_captioning_model(vocabulary_size=vocabulary_size, max_length=max_length, input_shape=Ximage_train[0].shape, weights_path=None)
    elif (int(command) == 2):
        model = neural_network_model.image_captioning_model_Bidirectional(vocabulary_size=vocabulary_size, max_length=max_length, input_shape=Ximage_train[0].shape, weights_path=None)
    elif (int(command) == 3):
        model = neural_network_model.image_captioning_model_Time_Distributed_Bidirectional(vocabulary_size=vocabulary_size, max_length=max_length, input_shape=Ximage_train[0].shape, weights_path=None)
    else:
        print("MODEL NOT SPECIFED, EXITING")
        sys.exit(0)

    hist = model.fit([Ximage_train, Xtext_train], ytext_train, batch_size=batch_size, verbose=1, epochs=epochs, validation_data=([Ximage_val, Xtext_val], ytext_val), shuffle=False)

    data_manager.display_predicted_captions(model=model, cnn_model=cnn_model, test_pic=5, img_test=coco_img_test, img_feat_test=img_features_test,
                                            img_cap_test=coco_img_cap_test, img_width=img_width, img_height=img_height,
                                            tokenizer=tokenizer, max_length=max_length)

    if (int(command) == 1):
        neural_network_model.save_weights(model=model, model_name_json="Model_Demo", weights_name="Weight_Demo")
        data_manager.tokenizer_object(tokenizer_obj=tokenizer, tokenizer_name="tokenizer_Demo", max_length_name="max_length_Demo", max_length=max_length)
        #neural_network_model.save_weights(model=model, model_name_json="Model_Basic", weights_name="Weight_Basic")
        #data_manager.tokenizer_object(tokenizer_obj=tokenizer, tokenizer_name="tokenizer_BASIC",max_length_name="max_length_Basic", max_length=max_length)
    elif (int(command) == 2):
        neural_network_model.save_weights(model=model, model_name_json="Model_Demo", weights_name="Weight_Demo")
        data_manager.tokenizer_object(tokenizer_obj=tokenizer, tokenizer_name="tokenizer_Demo", max_length_name="max_length_Demo", max_length=max_length)
        #neural_network_model.save_weights(model=model, model_name_json="Model_Bidirectional", weights_name="Weight_Bidirectional")
        #data_manager.tokenizer_object(tokenizer_obj=tokenizer, tokenizer_name="tokenizer_Bidirectional", max_length_name="max_length_Bidirectional", max_length=max_length)
    elif (int(command) == 3):
        neural_network_model.save_weights(model=model, model_name_json="Model_Demo", weights_name="Weight_Demo")
        data_manager.tokenizer_object(tokenizer_obj=tokenizer, tokenizer_name="tokenizer_Demo", max_length_name="max_length_Demo", max_length=max_length)
        #neural_network_model.save_weights(model=model, model_name_json="Model_TimeDistributed_Bidirectional", weights_name="Weight_TimeDistributed_Bidirectional")
        #data_manager.tokenizer_object(tokenizer_obj=tokenizer, tokenizer_name="tokenizer_TimeDistributed_Bidirectional", max_length_name="max_length_TimeDistributed_Bidirectional", max_length=max_length)
    data_manager.plot_statistics(hist)

if __name__ == '__main__':
    main(None)
