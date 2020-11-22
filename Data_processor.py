import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import cv2
import keras.backend as K
import pickle

from PIL import Image
from pprint import pprint
from tqdm import tqdm
from collections import Counter

from keras.models import Model
from keras import layers, models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical

class Data_processor:
    """
    # Load the json-files into python
    Param:
        j_file - Loads a .json file
    """
    def load_json(self, j_file):
        import json
        with open(j_file, 'r') as f:
            data = json.load(f)
            f.close()
        return data

    """
    # COCO_captions - Loading the image-captions into an array #
    Param:
        filename - a string datatype with the name of the file
    """
    def COCO_captions(self, filename):
        json_data = self.load_json(os.path.join('Dataset_COCO_captions', filename))
        unique_id = []
        img_id = []
        img_caption = []

        for i in range(len(json_data['annotations'])):
            unique_id.append(json_data['annotations'][i]['id'])
            img_id.append(json_data['annotations'][i]['image_id'])
            img_caption.append(json_data['annotations'][i]['caption'])
        return json_data, unique_id, img_id, img_caption

    """
    # COCO_images - Loading the image-id's into an array #
    Param:
        num_img - Number of images to load in
    """
    def COCO_images(self, num_img=100):
        image_array = []
        path = str('./Dataset_COCO_5k_images')
        count = 0
        for filename in tqdm(os.listdir(path)):
            if(count >= num_img):
                break
            image = Image.open(os.path.join('Dataset_COCO_5k_images', filename))
            image_array.append(filename)
            image.close()
            count += 1
        return image_array

    """
    # COCO_segmentation - Loading the segmentated image-id's into an array  #
    Param:
        num_img - number of images to load in
    """
    def COCO_images_segmentation(self, num_img=100):
        image_array_segmentation = []
        path = str('./Dataset_COCO_5k_images_segmentation')
        count = 0
        for filename in tqdm(os.listdir(path)):
            if(count >= num_img):
                break
            image = Image.open(os.path.join('Dataset_COCO_5k_images_segmentation', filename))
            image_array_segmentation.append(filename)
            image.close()
            count += 1
        return image_array_segmentation

    """
    # COCO_caption_ordered - Orders it so indexes are corresponding to each other #
    Param:
        img_array - an array consisting of image-filenames
        json_data - json data from .json file
        num_img - Number of images to load in
    """
    def COCO_caption_ordered(self, img_array, json_data, num_img=100):
        img_id = img_array[:]
        img_cap = []
        count = 0
        for images in tqdm(img_id):
            if(count >= num_img):
                break
            image_id = int(images[1:-4])
            annotation_id = self.find_image_id_annotation(image_id, json_data)
            img_cap_id = json_data['annotations'][annotation_id]['caption']
            img_cap.append(img_cap_id)
            count += 1
        return img_cap

    """
    # Removing some unecessary chars, punctuation and numbers to reduce complexity #
    Param:
        img_cap_array - an array consisting of captions
        num_img - Number of images to load in
    """
    def COCO_img_cap_clean(self, img_cap_array, num_img=100):
        img_cap = img_cap_array[:]
        img_cap_clean = []
        count = 0
        for elem in img_cap:
            if (count >= num_img):
                break
            elem = self.clean_single_char(elem)
            elem = self.clean_punctuation(elem)
            elem = self.clean_no_numbers(elem)
            img_cap_clean.append(elem)
            count += 1
        return img_cap_clean

    """
    # Remove punctuations from string #
    Param:
        text - a simple string
    """
    def clean_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        txt = text.translate(translator)
        return txt

    """
    # Remove single chars from string #
    Param:
        text - a simple string
    """
    def clean_single_char(self, text):
        txt = ""
        for word in text.split():
            if len(word) > 1:
                txt += " " + word
        return txt

    """
    # Remove numbers from string #
    Param:
        text - a simple string
    """
    def clean_no_numbers(self, text):
        result = ''.join([i for i in text if not i.isdigit()])
        return result

    """
    # Creating train-set and test-set with percentage and shuffling the dataset #
    Param:
        image_array - An image array consisting of image-filenames
        train_percentage - How much training data to return based on percent
        num_img - Number of images to load in
    """
    def COCO_train_test_set(self, image_array, train_percentage, num_img):
        orig = image_array[:]
        orig = np.random.permutation(orig)
        train_set = []
        test_set = []
        train_set_length = int(((len(image_array)/100)*train_percentage))
        train_set = orig[:train_set_length]
        test_set = orig[train_set_length:num_img]
        #train_set = np.random.permutation(train_set)
        #test_set = np.random.permutation(test_set)
        return train_set, test_set

    """
    # Find annotation-id given its image-id #
    Param:
        image_id - filename is the image id of the image
        json_data - json data from .json file
    """
    def find_image_id_annotation(self, image_id, json_data):
        path = str('./Dataset_COCO_5k_images')
        id_annotation = 0
        for i in json_data['annotations']:
            if(json_data['annotations'][id_annotation]['image_id'] == image_id):
                return id_annotation
            id_annotation += 1
        return 0

    """
    # Find annotation-id given its unique-id #
    Param:
        unique_id - id in the .json file 'NOT' filename for image reference
        json_data - json data from .json file
    """
    def find_unique_id_annotation(self, unique_id, json_data):
        id_annotation = 0
        for i in json_data['annotations']:
            if(json_data['annotations'][id_annotation]['id'] == unique_id):
                return id_annotation
            id_annotation += 1
        return 0

    """
    # Display the image with an unique-id #
    Param:
        unique_id - id in the .json file 'NOT' filename for image reference
        json_data - json data from .json file
    """
    def display_image_unique_id(self, unique_id, json_data):
        path = str('./Dataset_COCO_5k_images')

        img_id_annotation = 0
        count = 0
        for i in json_data['annotations']:
            if(json_data['annotations'][count]['id'] == unique_id):
                img_id_annotation = count
                break
            count += 1

        image = None # Image-object
        image_caption = None
        for filename in os.listdir(path):
            if(int(filename[1:-4]) == json_data['annotations'][img_id_annotation]['image_id']):
                image = Image.open(os.path.join('Dataset_COCO_5k_images', filename))
                image_caption = json_data['annotations'][img_id_annotation]['caption']
                print('Image-Filename: {}'.format(filename))
                print('Image-shape: {}'.format(image.size))
                print('Image-captioning: {}'.format(image_caption))
                plt.imshow(image)
                plt.show()
                image.close()
                break

    """
    # Display the image with an image-id #
    Param:
        image_id - filename is the image id of the image
        json_data - json data from .json file
    """
    def display_image_id(self, image_id, json_data):
        path = str('./Dataset_COCO_5k_images')

        img_id_annotation = 0
        count = 0
        for i in json_data['annotations']:
            if(json_data['annotations'][count]['image_id'] == image_id):
                img_id_annotation = count
                break
            count += 1

        image = None # Image-object
        image_caption = None
        for filename in os.listdir(path):
            if(int(filename[1:-4]) == json_data['annotations'][img_id_annotation]['image_id']):
                image = Image.open(os.path.join('Dataset_COCO_5k_images', filename))
                image_caption = json_data['annotations'][img_id_annotation]['caption']
                print('Image-Filename: {}'.format(filename))
                print('Image-shape: {}'.format(image.size))
                print('Image-captioning: {}'.format(image_caption))
                plt.imshow(image)
                plt.show()
                image.close()
                break

    """
    # Preprocessing - Converting every images in an array to a proper input-shape for the CNN-model #
    Param:
        image_array - An array consisting of filenames to iamges
        height - target height
        width - target width
        path - designated path to the images in the image array
    """
    def convert_img_to_array(self, image_array, height, width, path):
        x = image_array[:]

        new_img_array = []
        for images in tqdm(x):
            orig_img = image.load_img(os.path.join(path, images), target_size=(height, width))
            img_data = image.img_to_array(orig_img)
            #img_data = np.expand_dims(img_data, axis=0)
            new_img_array.append(img_data)

        return new_img_array

    """
    # Feature-extraction using CNN-model #
    Param:
        image_array - An array consisting of filenames to images
        height - target height
        width - target width
        path - designated path to the images in the image img_array
        CNN_model - model for the Convolutional neural network
    """
    def img_feat(self, image_array, height, width, path, CNN_model):
        pred_model = CNN_model
        pred_model = Model(inputs=pred_model.inputs, outputs=pred_model.layers[-2].output)
        #pred_model.summary()
        x = image_array[:]

        img_feat_dict = []
        for images in tqdm(x):
            orig_img = image.load_img(os.path.join(path, images), target_size=(height, width))
            img_data = image.img_to_array(orig_img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            img_features = pred_model.predict(img_data)
            img_feat_dict.append(img_features)
        return img_feat_dict

    """
    # Counting amount of unique words and total words of the images in total #
    Param:
        img_cap_array - An array consisting of captions for the images
    """
    def text_processing(self, img_cap_array):
        vocabulary = []
        x = img_cap_array[:]
        for txt in tqdm(x):
            vocabulary.extend(txt.split())
        print('Total words: %d' % (len(vocabulary)+1))

        new_vocabulary = list(set(vocabulary))
        print('Vocabulary size: %d' % (len(new_vocabulary)+1))
        return vocabulary, new_vocabulary, (len(new_vocabulary)+1)

    """
    # Adding start and end token at the sequence #
    Param:
        img_cap_array - An array consisting of captions for the images
    """
    def start_end_seq_token(self, img_cap_array):
        tmp = []
        x = img_cap_array[:]
        for elem in x:
            elem = '<startseq> ' + elem + ' <endseq>'
            tmp.append(elem)
        return tmp

    """
    # Integers to represents chars #
    Param:
        img_cap_array - An array consisting of captions for the images
        vocabulary_size - the total size of the vocabulary, amount of unique words
    """
    def tokenize_char_to_int(self, img_cap_array, vocabulary_size):
        x = img_cap_array[:]

        tokenizer = Tokenizer(num_words=vocabulary_size)
        tokenizer.fit_on_texts(x)
        tokenized_text = tokenizer.texts_to_sequences(x)
        N_length = np.max([len(text) for text in tokenized_text])

        return tokenized_text, tokenizer, N_length

    """
    # Detokenizing from integers to chars, reversing the integers back to words #
    Param:
        dict_tokenizer - Tokenizer object
        img_cap_tokenized - An array consisting of tokenized integers
    """
    def detokenize_int_to_char(self, dict_tokenizer, img_cap_tokenized):
        decrypted_caption = []
        for integer in img_cap_tokenized:
            for word, index in dict_tokenizer.word_index.items():
                    if (index == integer):
                               decrypted_caption.append(word)
                    #print(integer)
        return decrypted_caption

    """
    # Save tokenizer_object for the specific model as every language model has their own corpus #
    Param:
        tokenizer_obj - Tokenizer object
        tokenizer_name - Name of the tokenizer object
        max_length_name - Name of the output file to store max_length
        max_length - a number specifying the max_length of a sequence
    """
    def tokenizer_object(self, tokenizer_obj, tokenizer_name, max_length_name ,max_length):
        with open(str(tokenizer_name) + '.pickle', 'wb') as fp:
            pickle.dump(tokenizer_obj, fp, protocol=pickle.HIGHEST_PROTOCOL)

        length_file = open(str(max_length_name) + ".txt", "w")
        length_file.write(str(max_length))
        length_file.close()

    """
    # Find the word based on integer number #
    Param:
        integer - any arbitrary number
        tokenizer - tokenizer object
    """
    def word_for_id(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    """
    # Create sequences where caption and image features are combined. #
    Param:
        img_cap_array_tokenized - An array consisting of captions for the images, but the captions are encoded/tokenized
        img_feature_array - An array consisting of feature-vectors for the images
        vocab_size - Total vocabulary size, ie a number
        max_len - max length of the sequence
    """
    def preprocessing(self,img_cap_array_tokenized,img_feature_array,vocab_size,max_len):
        N = len(img_cap_array_tokenized)
        print("# captions/images = {}".format(N))

        Xtext,Ximage, ytext = [],[],[]
        for text, image in zip(img_cap_array_tokenized, img_feature_array):
            for i in range(1, len(text)):
                in_text, out_text = text[:i], text[i]
                in_text = pad_sequences([in_text], maxlen=max_len, padding='post').flatten()
                out_text = to_categorical(out_text, num_classes=vocab_size)

                Xtext.append(in_text)
                Ximage.append(image)
                ytext.append(out_text)

        Xtext  = np.array(Xtext)
        Ximage = np.array(Ximage)
        ytext  = np.array(ytext)
        #print(" {} {} {}".format(Xtext.shape,Ximage.shape, ytext.shape))
        return(Xtext,Ximage,ytext)

    """
    # Predict the caption for the requested image. #
    Param:
        neural_model - The language model
        tokenizer - Tokenizer object
        photo - a feature vector which describes the image
        max_length - max length of the sequence
    """
    def predict_caption(self, neural_model, tokenizer, photo, max_length):
        in_text = '<startseq> '
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = neural_model.predict([photo, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self.word_for_id(yhat, tokenizer)
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
        test_pic - A number to specify how many images to fit into plot
        img_test - An array consisting of images to test
        img_feat_test - An array consisting of feature-vector for images
        img_cap_test - An array consisting of actual caption for the images
        img_width - target width
        img_height - target height
        tokenizer - Tokenizer object
        max_length - max length of the sequence
    """
    def display_predicted_captions(self,model,cnn_model, test_pic, img_test, img_feat_test, img_cap_test, img_width, img_height, tokenizer, max_length):
        test_pictures = test_pic
        count = 1

        fig = plt.figure(figsize=(30,30))
        for filename, image_feature, img_cap in zip(img_test[:test_pictures], img_feat_test[:test_pictures], img_cap_test[:test_pictures]):
            image_load = load_img(os.path.join('Dataset_COCO_5k_images', filename), target_size=(img_width, img_height,3))
            ax = fig.add_subplot(test_pictures,3,count,xticks=[],yticks=[])
            ax.imshow(image_load)
            count += 1

            ax = fig.add_subplot(test_pictures,3,count,xticks=[],yticks=[])
            #heatmap_VGG19 = self.dispaly_heatmap(model=cnn_model, image_name=filename, path=str('./Dataset_COCO_5k_images'), layer_name_activations="block5_conv3")
            heatmap_ResNet50 = self.display_heatmap(model=cnn_model, image_name=filename, path=str('./Dataset_COCO_5k_images'), layer_name_activations="activation_49")
            ax.imshow(heatmap_ResNet50)
            count += 1

            caption = self.predict_caption(neural_model=model, tokenizer=tokenizer, photo=image_feature, max_length=max_length)
            ax = fig.add_subplot(test_pictures,3,count)
            plt.axis('off')
            ax.plot()
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.text(0,0.7,"PREDICTED CAPTION: {}".format(caption),fontsize=7)
            ax.text(0,0.5,"ACTUAL CAPTION: {}".format(img_cap), fontsize=7)
            count += 1
        plt.show()

    """
    # Removing 'startseq' and 'endseq' for cleaner sentences
    Param:
        caption_text - A string of text that contains <startseq> and <endseq>
    """
    def remove_start_end_seq(self, caption_text):
        orig = caption_text[:]
        length = len(orig)
        orig = orig[8:] #orig[8:-6]
        return orig

    """
    'HEATMAP - Keras' - Retrieves the activation functions from the CNN-model for a specific layer.
    Inspired Code taken from: http://www.hackevolve.com/where-cnn-is-looking-grad-cam/
    Credits: Saideep
    """
    def display_heatmap(self, model, image_name, path, layer_name_activations):
        model_testing = model
        img = image.load_img(os.path.join(path, image_name), target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        pred = model_testing.predict(x)
        class_idx = np.argmax(pred[0])
        class_output = model_testing.output[:, class_idx]
        last_conv_layer = model_testing.get_layer(layer_name_activations)#block5_covv3
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

    """
    # Plotting the training statistics of the neural network model #
    Param:
        history_model - fit function from keras.fit
    """
    def plot_statistics(self, history_model):
        plt.subplot(2,1,1)
        plt.plot(history_model.history['acc'])
        plt.plot(history_model.history['val_acc'])
        plt.title('Accuracy of training phase:')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'], loc='upper right')

        plt.subplot(2,1,2)
        plt.plot(history_model.history['loss'])
        plt.plot(history_model.history['val_loss'])
        plt.title('Loss of training phase:')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
        plt.show()
