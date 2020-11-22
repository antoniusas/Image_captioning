import matplotlib.pyplot as plt
import numpy as np

from keras import losses
from keras import regularizers, initializers
from keras.utils import to_categorical
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, Bidirectional
from keras.layers import Input
from keras.utils import plot_model
from keras.models import Model, model_from_json
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers.merge import concatenate
from keras.layers.pooling import GlobalMaxPooling2D, AveragePooling2D
from keras import regularizers, optimizers

class Neural_network_model:
    # Constructor, essential params
    def __init__(self):
        pass
    # Convolutional Network - Model used for feature-extraction from image - CUSTOM made
    """
    "Orginally planned to use this custom-made CNN-model for feature-extraction by training it on my own. But this is however never used. "
    Param:
        input_shape - input shape for the image
        num_features - amount of units for the second last layer
        num_features_class - amount of units for the last layer
        num_filter1 - Number of filters for the two first conv-layers
        num_filter2 - Number of filters for the two middle conv-layers
        num_filter3 - Number of filters for the two last conv-layers
        img_strides1 - Stride-shifting for the two first conv-layers
        img_strides2 - Stride-shifting for the two middle conv-layers
        img_strides3 - Stride-shifting for the two last conv-layers
        weights_path - Specify the path of the weight file to load the weights
    """
    def conv2d_neural_model(self, input_shape, num_features, num_features_class,num_filter1, num_filter2, num_filter3, img_strides1, img_strides2, img_strides3, weights_path=None):
        conv_model = Sequential()
        conv_model.add(Conv2D(filters=num_filter1, kernel_size=(3,3), strides=img_strides1, padding='same', activation='relu', input_shape=input_shape))
        conv_model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        conv_model.add(Dropout(0.0))
        conv_model.add(Conv2D(filters=num_filter1, kernel_size=(3,3), strides=img_strides1, padding='same', activation='relu'))
        conv_model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        conv_model.add(Dropout(0.0))
        conv_model.add(Conv2D(filters=num_filter2, kernel_size=(3,3), strides=img_strides2, padding='same', activation='relu'))
        conv_model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        conv_model.add(Dropout(0.0))
        conv_model.add(Conv2D(filters=num_filter2, kernel_size=(3,3), strides=img_strides2, padding='same', activation='relu'))
        conv_model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        conv_model.add(Dropout(0.0))
        conv_model.add(Conv2D(filters=num_filter3, kernel_size=(3,3), strides=img_strides3, padding='same', activation='relu'))
        conv_model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        conv_model.add(Dropout(0.0))
        conv_model.add(Conv2D(filters=num_filter3, kernel_size=(3,3), strides=img_strides3, padding='same', activation='relu'))
        conv_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        conv_model.add(Dropout(0.0))
        conv_model.add(Flatten())
        conv_model.add(Dense(num_features, activation='relu'))
        conv_model.add(Dropout(0.5))
        conv_model.add(Dense(num_features, activation='relu'))
        conv_model.add(Dropout(0.5))
        conv_model.add(Dense(num_features_class, activation='softmax'))
        # Training the model first with classification
        if weights_path:
            model.load_weights(weights_path)
            return model
        conv_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        conv_model.summary()
        return conv_model

    """
    # Language-Model I, standard LSTM #
    Param:
        vocabulary_size - Size of dictionary
        max_length - maximum length of the sequence for the model
        input_shape - shape for feature-vector
        weights_path - Specify the path of the weight file to load the weights
    """
    def basic_image_captioning_model(self, vocabulary_size, max_length, input_shape, weights_path=None):
        dim_embedding = 512
        # Image features model - Layer #
        input_image = Input(shape=input_shape)
        fimage1 = Dropout(0.5)(input_image)
        fimage2 = Dense(512,activation='relu',name="ImageFeature", kernel_regularizer=regularizers.l2(0.01))(input_image)  #(fimage1)
        # Sequence model with Embedding - Layer #
        input_txt = Input(shape=(max_length,))
        ftxt1 = Embedding(vocabulary_size, dim_embedding,input_length=max_length, mask_zero=True)(input_txt)
        ftxt3 = LSTM(256,name="CaptionFeature", return_sequences=True)(ftxt1) #(ftxt2)
        ftxt5 = LSTM(256,name="CaptionFeature1")(ftxt3)
        # Sequence and Image feature model combined for decoder #
        decoder = concatenate([ftxt5,fimage2])
        decoder1 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(decoder)
        decoder2 = Dropout(0.5)(decoder1)
        output = Dense(vocabulary_size,activation='softmax')(decoder2)
        model = Model(inputs=[input_image, input_txt],outputs=output)
        # Saving the weights if weights_path == true #
        if weights_path:
            model.load_weights(weights_path)
            return model
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        model.summary()
        return model

    """
    # Language-Model II, Bidirectional LSTM #
    Param:
        vocabulary_size - Size of dictionary
        max_length - maximum length of the sequence for the model
        input_shape - shape for feature-vector
        weights_path - Specify the path of the weight file to load the weights
    """
    def image_captioning_model_Bidirectional(self, vocabulary_size, max_length, input_shape, weights_path=None):
        dim_embedding = 512
        # Image features model - Layer #
        input_image = Input(shape=input_shape)
        fimage1 = Dense(512,activation='relu',name="ImageFeature", kernel_regularizer=regularizers.l2(0.01))(input_image)  #(fimage1)
        fimage2 = Dropout(0.5)(fimage1)
        ## Sequence model with Embedding - Layer #
        input_txt = Input(shape=(max_length,))
        ftxt1 = Embedding(vocabulary_size, dim_embedding, mask_zero=True)(input_txt)
        ftxt3 = Bidirectional(LSTM(256,name="CaptionFeature",return_sequences=True))(ftxt1)
        ftxt5 = Bidirectional(LSTM(256,name="CaptionFeature2"))(ftxt3)
        # Sequence and Image feature model combined for decoder #
        decoder = concatenate([ftxt5,fimage2])
        decoder1 = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.01))(decoder)
        decoder2 = Dropout(0.5)(decoder1)
        output = Dense(vocabulary_size,activation='softmax')(decoder2)
        model = Model(inputs=[input_image, input_txt],outputs=output)
        # Saving the weights if weights_path == true #
        if weights_path:
            model.load_weights(weights_path)
            return model
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        model.summary()
        return model

    """
    # Language-Model III, Time-Distributed-Bidirectional LSTM #
    Param:
        vocabulary_size - Size of dictionary
        max_length - maximum length of the sequence for the model
        input_shape - shape for feature-vector
        weights_path - Specify the path of the weight file to load the weights
    """
    def image_captioning_model_Time_Distributed_Bidirectional(self, vocabulary_size, max_length, input_shape, weights_path=None):
        dim_embedding = 512
        # Image features model - Layer #
        input_image = Input(shape=input_shape)
        fimage1 = Dense(512, activation='relu',name="ImageFeature", kernel_regularizer=regularizers.l2(0.01))(input_image)
        fimage2 = Dropout(0.5)(fimage1)
        fimage3 = RepeatVector(max_length)(fimage2)
        # Sequence model with Embedding - Layer #
        input_txt = Input(shape=(max_length,))
        ftxt1 = Embedding(vocabulary_size, dim_embedding,input_length=max_length ,mask_zero=True)(input_txt)
        ftxt2 = Bidirectional(LSTM(256,name="CaptionFeature", return_sequences=True))(ftxt1)
        ftxt3 = Bidirectional(LSTM(256,name="CaptionFeature2", return_sequences=True))(ftxt2)
        ftxt4 = TimeDistributed(Dense(256))(ftxt3)
        # Sequence and Image feature model combined for decoder #
        decoder = concatenate([ftxt4,fimage3])
        decoder1 = LSTM(256, activation='relu')(decoder)
        decoder2 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(decoder1)
        decoder3 = Dropout(0.5)(decoder2)
        output = Dense(vocabulary_size,activation='softmax')(decoder3)
        model = Model(inputs=[input_image, input_txt],outputs=output)
        # Saving the weights if weights_path == true #
        if weights_path:
            model.load_weights(weights_path)
            return model
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        model.summary()
        return model

    """
    # Saving the weights #
    Param:
        model - load any neural network model
        model_name_json - name the .json file
        weights_name - name the .h5 file
    """
    def save_weights(self, model, model_name_json ,weights_name):
        model_json = model.to_json()
        with open(str(model_name_json) + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(str(weights_name) + ".h5")

    """
    # Load the Weights #
    Param:
        model_name_json - name the .json file
        weights_name - name the .h5 file
    """
    def load_weights(self, model_name_json, weights_name):
        json_file = open(str(model_name_json) + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(str(weights_name)+".h5")
        loaded_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        return loaded_model
