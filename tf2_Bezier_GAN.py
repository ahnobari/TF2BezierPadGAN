import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from tqdm.autonotebook import tqdm, trange
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models import Bezier_Discriminator, Bezier_Generator, Surrogate_Lift2Drag, Bezier_PaDGAN


if __name__ == '__main__':
    model = Bezier_PaDGAN()
    X_train = np.load('data/xs_train.npy').astype('float32') 
    X_test = np.load('data/xs_test.npy').astype('float32') 
    Y_train = np.load('data/ys_train.npy').astype('float32') 
    Y_test = np.load('data/ys_test.npy').astype('float32') 
    Y = np.concatenate((Y_train,Y_test))
    min_y = np.min(Y)
    max_y = np.max(Y)
    Y_train = (Y_train - min_y)/(max_y - min_y)
    Y_test = (Y_test - min_y)/(max_y - min_y)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    Y_train = np.expand_dims(Y_train, axis=-1)
    Y_test = np.expand_dims(Y_test, axis=-1)

    
    model.load_model(name = 'surrogate_trained', model="surrogate")
    print("Surrogate Model Loaded")
    print("Starting Training of BezierPadGAN")
    model.train_GAN(X_train, steps=10000, batch_size=32, disc_lr=1e-4, gen_lr=1e-4)
    print("Training Completed Saving...")
    model.save_model(name = 'Trained_GAN')
    print("Trained GAN Saved")
    print('generating Samples...')
    c = tf.random.uniform((32,5), minval = 0., maxval = 1., dtype=tf.float32)
    z = tf.random.normal((32,10), stddev = 0.5, dtype=tf.float32)
    sample = model.generator(c,z)[0]
    sample = tf.squeeze(sample)
    fig = make_subplots(rows=2, cols=2)
    fig.add_trace(go.Scatter(x=sample[0,:,0],y=sample[0,:,1]), row=1, col=1)
    fig.add_trace(go.Scatter(x=sample[1,:,0],y=sample[1,:,1]), row=1, col=2)
    fig.add_trace(go.Scatter(x=sample[2,:,0],y=sample[2,:,1]), row=2, col=1)
    fig.add_trace(go.Scatter(x=sample[3,:,0],y=sample[3,:,1]), row=2, col=2)
    fig.update_xaxes(title_text='x')
    fig.update_yaxes(title_text='y',scaleanchor='x',scaleratio=1)
    fig.show()