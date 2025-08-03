'''
Created on June 3rd 2025

@author: Bella Chung

Based on https://github.com/csong27/membership-inference/blob/master/classifier.py, 
         https://github.com/AhmedSalem2/ML-Leaks/blob/master/classifier.py

'''

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras import layers

def get_cnn_model(n_in, n_hidden, n_out):
    '''
    Constructs a Convolutional Neural Network (CNN) model.
    Added dropout layers after max pooling to prevent overfitting.
    '''
    model = keras.Sequential([
        # Input layer expects shape (height, width, channels) for Conv2D.
        # n_in[1:] because n_in[0] is batch size.
        layers.InputLayer(input_shape=n_in[1:]),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                      kernel_initializer='glorot_uniform'),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                      kernel_initializer='glorot_uniform'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu',
                      kernel_initializer='glorot_uniform'),
        layers.Conv2D(64, (3, 3), activation='relu',
                      kernel_initializer='glorot_uniform'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),  # Flatten the output of convolutional layers for dense layers
        layers.Dense(n_hidden, activation='tanh', name='features_layer'),
        layers.Dropout(0.50),
        layers.Dense(n_out, activation='softmax', name='output_softmax')
    ])
    return model

def get_cnn_features(trained_model, input_shape):
    # Create a new Input layer for the feature extraction model
    input_tensor = keras.Input(shape=input_shape)
    # Pass this input tensor through the layers of the trained_model
    # up to the features_layer
    x = input_tensor

    for layer in trained_model.layers:
        # Stop when we reach the features_layer
        if layer.name == 'features_layer':
            features_output = layer(x)
            break
        # Otherwise, pass through the layer
        x = layer(x)

    # Create the new functional model
    features_model = keras.Model(inputs=input_tensor, outputs=features_output)
    return features_model

def get_nn_model(n_in, n_hidden, n_out):
    '''
    Constructs a vanilla, Multi-layer Perceptron (MLP) model.
    '''
    model = keras.Sequential([
        # n_in[1] because n_in[0] is batch size
        layers.InputLayer(input_shape=(n_in[1],)),
        layers.Dense(n_hidden, activation='tanh'),
        layers.Dense(n_out, activation='softmax')
    ])
    return model

def get_softmax_model(n_in, n_out):
    '''
    Constructs a Softmax logistic regression model.
    '''
    model = keras.Sequential([
        layers.InputLayer(input_shape=(n_in[1],)),
        layers.Dense(n_out, activation='softmax')
    ])
    return model

def train_model(train_x, train_y, test_x=None, test_y=None, model_type='cnn', n_hidden=128, batch_size=100, epochs=100, learning_rate=0.001, l2_reg_strength=0):
    '''
    If test_x and test_y are not given, early stopping is not implemented.

    Args:
        train_x: train features
        train_y: train labels
        test_x: test features (used as validation for early stopping)
        test_y: test labels (used as validation for early stopping)
        model_type: cnn, nn, soft
        n_hidden: number of hidden layer nodes
        batch_size: batch size
        epochs: number of epochs
        learning_rate: learning rate
        l2_reg_strength: strength of l2 penalty on model loss

    Returns:
        keras.Sequential: trained_model
    '''
    n_in = train_x.shape
    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)

    if model_type == 'cnn':
        model = get_cnn_model(n_in, n_hidden, n_out)
    elif model_type == 'nn':
        model = get_nn_model(n_in, n_hidden, n_out)
    else:
        model = get_softmax_model(n_in, n_out)

    model.summary()

    # --- Explicitly add L2 regularization loss ---
    # Iterate over all trainable weights (kernels) and add their L2 penalty to the model's losses.
    if l2_reg_strength > 0: # Check if regularization is enabled
        print(f"Applying L2 regularization with strength: {l2_reg_strength}")
        for weight in model.trainable_weights:
            # Typically, we only regularize kernel weights, not bias weights.
            if "kernel" in weight.name: 
                model.add_loss(l2_reg_strength * tf.reduce_sum(tf.square(weight)))
    
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy', # Use sparse_categorical_crossentropy for integer labels
                  metrics=['accuracy'])
    
    if test_x is None or test_y is None:
        print('Training...')
        model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1) # Set to 1 or 2 for more verbose output; 0 is silent
    else:
        # Define EarlyStopping callback
        early_stopping_monitor = EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            min_delta=0.001,     # Minimum change to qualify as an improvement
            patience=10,         # Number of epochs with no improvement after which training will be stopped
            verbose=1,
            mode='min',          # 'min' means stop when the monitored quantity stops decreasing
            restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity.
        )

        print('Training with Early Stopping...')
        model.fit(train_x, train_y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(test_x, test_y), # Crucial for early stopping
                callbacks=[early_stopping_monitor]
                )
    
    print('Epochs completed.')

    return model

def model_report(model, test_x, test_y, batch_size=100):
    '''
    Performs testing on the trained model using accuracy metric.
    '''
    test_pred_probs = model.predict(test_x, batch_size=batch_size)
    pred_y = np.argmax(test_pred_probs, axis=1) # This is the final prediction for testing

    print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    print('More detailed results:')
    print(classification_report(test_y, pred_y))