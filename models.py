import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from tqdm.autonotebook import tqdm, trange

class Residual_Block(keras.layers.Layer):
    def __init__(self, channels, kernel_size, kernel_initializer, kernel_regularizer, downsample=False):
        super(Residual_Block, self).__init__()
        self.downsample = downsample
        self.batch_norm_1 = keras.layers.BatchNormalization(momentum=0.9)
        self.activation_1 = keras.layers.LeakyReLU(alpha=0.2)
        if downsample:
            self.conv_1 = keras.layers.Conv2D(channels, kernel_size, strides=(2,1), padding='same', 
                                              kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
            self.conv_1_1 = keras.layers.Conv2D(channels, kernel_size = 1, strides=(2,1), padding='same', 
                                              kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        else:
            self.conv_1 = keras.layers.Conv2D(channels, kernel_size, strides=1, padding='same', 
                                              kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        
        self.conv_2 = keras.layers.Conv2D(channels, kernel_size = kernel_size, strides=1, padding='same', 
                                          kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.batch_norm_2 = keras.layers.BatchNormalization(momentum=0.9)
        self.activation_2 = keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs, training = True):
        x_init = inputs
        x = self.batch_norm_1(inputs, training = training)
        x = self.activation_1(x)
        
        if self.downsample:
            x_init = self.conv_1_1(x_init)
        
        x = self.conv_1(x)
        x = self.batch_norm_2(x, training = training)
        x = self.activation_2(x)
        x = self.conv_2(x)
        
        x = x + x_init
        
        return x

class Bezier_Discriminator(keras.Model):
    def __init__(self, latent_dim, EPSILON):
        super(Bezier_Discriminator, self).__init__()
        
        self.disc_depth = 64
        self.disc_dropout = 0.4
        self.disc_kernel_size = (4,2)
        self.latent_dim = latent_dim
        self.EPSILON = EPSILON
        
        self.disc_layer_1 = keras.layers.Conv2D(self.disc_depth,self.disc_kernel_size, strides=(2,1), padding='same')
        self.disc_batchnorm_1 = keras.layers.BatchNormalization(momentum=0.9)
        self.disc_activation_1= keras.layers.LeakyReLU(alpha=0.2)
        self.disc_dropout_1 = keras.layers.Dropout(self.disc_dropout)
        
        self.disc_layer_2 = keras.layers.Conv2D(self.disc_depth * 2,self.disc_kernel_size, strides=(2,1), padding='same')
        self.disc_batchnorm_2 = keras.layers.BatchNormalization(momentum=0.9)
        self.disc_activation_2= keras.layers.LeakyReLU(alpha=0.2)
        self.disc_dropout_2 = keras.layers.Dropout(self.disc_dropout)
        
        self.disc_layer_3 = keras.layers.Conv2D(self.disc_depth * 4,self.disc_kernel_size, strides=(2,1), padding='same')
        self.disc_batchnorm_3 = keras.layers.BatchNormalization(momentum=0.9)
        self.disc_activation_3= keras.layers.LeakyReLU(alpha=0.2)
        self.disc_dropout_3 = keras.layers.Dropout(self.disc_dropout)
        
        self.disc_layer_4 = keras.layers.Conv2D(self.disc_depth * 8,self.disc_kernel_size, strides=(2,1), padding='same')
        self.disc_batchnorm_4 = keras.layers.BatchNormalization(momentum=0.9)
        self.disc_activation_4 = keras.layers.LeakyReLU(alpha=0.2)
        self.disc_dropout_4 = keras.layers.Dropout(self.disc_dropout)
        
        self.disc_layer_5 = keras.layers.Conv2D(self.disc_depth * 16,self.disc_kernel_size, strides=(2,1), padding='same')
        self.disc_batchnorm_5 = keras.layers.BatchNormalization(momentum=0.9)
        self.disc_activation_5 = keras.layers.LeakyReLU(alpha=0.2)
        self.disc_dropout_5 = keras.layers.Dropout(self.disc_dropout)
        
        self.disc_layer_6 = keras.layers.Conv2D(self.disc_depth * 32,self.disc_kernel_size, strides=(2,1), padding='same')
        self.disc_batchnorm_6 = keras.layers.BatchNormalization(momentum=0.9)
        self.disc_activation_6 = keras.layers.LeakyReLU(alpha=0.2)
        self.disc_dropout_6 = keras.layers.Dropout(self.disc_dropout)
        
        self.disc_layer_7 =  keras.layers.Flatten()
        self.disc_layer_8 = keras.layers.Dense(1024)
        self.disc_batchnorm_8 = keras.layers.BatchNormalization(momentum=0.9)
        self.disc_activation_8 = keras.layers.LeakyReLU(alpha=0.2)
        
        self.d_predict = keras.layers.Dense(1)
        
        self.d_q = keras.layers.Dense(128)
        self.d_q_batchnorm_9 = keras.layers.BatchNormalization(momentum=0.9)
        self.d_q_activation = keras.layers.LeakyReLU(alpha=0.2)
        
        self.d_q_mean = keras.layers.Dense(self.latent_dim)
        
        self.d_q_logstd = keras.layers.Dense(self.latent_dim)
    
    def call(self, inputs, training = True):
        
        x = inputs
        x = self.disc_layer_1(x)
        x = self.disc_batchnorm_1(x, training = training)
        x = self.disc_activation_1(x)
        x = self.disc_dropout_1(x, training = training)

        x = self.disc_layer_2(x)
        x = self.disc_batchnorm_2(x, training = training)
        x = self.disc_activation_2(x)
        x = self.disc_dropout_2(x, training = training)
        
        x = self.disc_layer_3(x)
        x = self.disc_batchnorm_3(x, training = training)
        x = self.disc_activation_3(x)
        x = self.disc_dropout_3(x, training = training)
        
        x = self.disc_layer_4(x)
        x = self.disc_batchnorm_4(x, training = training)
        x = self.disc_activation_4(x)
        x = self.disc_dropout_4(x, training = training)
        
        x = self.disc_layer_5(x)
        x = self.disc_batchnorm_5(x, training = training)
        x = self.disc_activation_5(x)
        x = self.disc_dropout_5(x, training = training)
        
        x = self.disc_layer_6(x)
        x = self.disc_batchnorm_6(x, training = training)
        x = self.disc_activation_6(x)
        x = self.disc_dropout_6(x, training = training)
        
        x = self.disc_layer_7(x)
        x = self.disc_layer_8(x)
        x = self.disc_batchnorm_8(x, training = training)
        x = self.disc_activation_8(x)

        d = self.d_predict(x)
        
        q = self.d_q(x)
        q = self.d_q_batchnorm_9(q, training = training)
        q = self.d_q_activation(q)

        q_mean = self.d_q_mean(q)
        q_logstd = self.d_q_logstd(q)
        q_logstd = tf.math.maximum(q_logstd, -16)
        q_mean = tf.reshape(q_mean, (-1, 1, self.latent_dim))
        q_logstd = tf.reshape(q_logstd, (-1, 1, self.latent_dim))
        q = tf.concat([q_mean, q_logstd], axis=1) # batch_size x 2 x latent_dim

        return d, q

class Bezier_Generator(keras.Model):
    def __init__(self, latent_dim, noise_dim, bezier_degree, X_shape, EPSILON = 1e-7):
        super(Bezier_Generator, self).__init__()
        
        self.bezier_degree = bezier_degree
        self.depth_cpw = 32*8
        self.dim_cpw = int((self.bezier_degree+1)/8)
        self.cpw_kernel_size = (4,3)
        self.EPSILON = EPSILON
        self.X_shape = X_shape
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        
        self.cpw_layer_1 = keras.layers.Dense(1024, input_shape = (self.latent_dim + self.noise_dim,))
        self.cpw_batchnorm_1 = keras.layers.BatchNormalization(momentum=0.9)
        self.cpw_activation_1 = keras.layers.LeakyReLU(alpha=0.2)
        self.cpw_layer_2 = keras.layers.Dense(self.dim_cpw * 3 * self.depth_cpw)
        self.cpw_batchnorm_2 = keras.layers.BatchNormalization(momentum=0.9)
        self.cpw_activation_2 = keras.layers.LeakyReLU(alpha=0.2)
        self.cpw_layer_3 = keras.layers.Conv2DTranspose(int(self.depth_cpw/2), self.cpw_kernel_size, strides=(2,1), padding='same')
        self.cpw_batchnorm_3 = keras.layers.BatchNormalization(momentum=0.9)
        self.cpw_activation_3 = keras.layers.LeakyReLU(alpha=0.2)
        self.cpw_layer_4 = keras.layers.Conv2DTranspose(int(self.depth_cpw/4), self.cpw_kernel_size, strides=(2,1), padding='same')
        self.cpw_batchnorm_4 = keras.layers.BatchNormalization(momentum=0.9)
        self.cpw_activation_4 = keras.layers.LeakyReLU(alpha=0.2)
        self.cpw_layer_5 = keras.layers.Conv2DTranspose(int(self.depth_cpw/8), self.cpw_kernel_size, strides=(2,1), padding='same')
        self.cpw_batchnorm_5 = keras.layers.BatchNormalization(momentum=0.9)
        self.cpw_activation_5 = keras.layers.LeakyReLU(alpha=0.2)
        
        self.cp_layer = keras.layers.Conv2D(1, (1,2), padding='valid', input_shape = (self.dim_cpw * 8, 3, int(self.depth_cpw/8)))
        self.cp_activation = keras.layers.Activation(keras.activations.tanh)
        
        self.w_layer = keras.layers.Conv2D(1, (1,3), padding='valid', input_shape = (self.dim_cpw * 8, 3, int(self.depth_cpw/8)))
        self.w_activation = keras.layers.Activation(keras.activations.sigmoid)
        
        self.db_layer_1 = keras.layers.Dense(1024)
        self.db_batch_norm_1 = keras.layers.BatchNormalization(momentum=0.9)
        self.db_activation_1 = keras.layers.LeakyReLU(alpha=0.2)
        self.db_layer_2 = keras.layers.Dense(256)
        self.db_batch_norm_2 = keras.layers.BatchNormalization(momentum=0.9)
        self.db_activation_2 = keras.layers.LeakyReLU(alpha=0.2)
        self.db_layer_3 = keras.layers.Dense(self.X_shape[0] - 1)
        self.db_activation_3 = keras.layers.Softmax()
        

    def call(self, c, z, training = True):
        if self.noise_dim == 0:
            cz = c
        else:
            cz = tf.concat([c, z], axis=-1)
            
        cpw = self.cpw_layer_1(cz)
        cpw = self.cpw_batchnorm_1(cpw, training = training)
        cpw = self.cpw_activation_1(cpw)
        cpw = self.cpw_layer_2(cpw)
        cpw = self.cpw_batchnorm_2(cpw, training = training)
        cpw = self.cpw_activation_2(cpw)
        cpw = tf.reshape(cpw, (-1, self.dim_cpw, 3, self.depth_cpw))
        cpw = self.cpw_layer_3(cpw)
        cpw = self.cpw_batchnorm_3(cpw, training = training)
        cpw = self.cpw_activation_3(cpw)
        cpw = self.cpw_layer_4(cpw)
        cpw = self.cpw_batchnorm_4(cpw, training = training)
        cpw = self.cpw_activation_4(cpw)
        cpw = self.cpw_layer_5(cpw)
        cpw = self.cpw_batchnorm_5(cpw, training = training)
        cpw = self.cpw_activation_5(cpw)
        
        # control points
        cp = self.cp_layer(cpw)
        cp = self.cp_activation(cp)
        cp = tf.squeeze(cp, axis = -1)

        # weights
        w = self.w_layer(cpw)
        w = self.w_activation(w)
        w = tf.squeeze(w, axis = -1)

        # Parameters at data points
        db = self.db_layer_1(cz)
        db = self.db_batch_norm_1(db, training = training)
        db = self.db_activation_1(db)
        db = self.db_layer_2(db)
        db = self.db_batch_norm_2(db, training = training)
        db = self.db_activation_2(db)
        db = self.db_layer_3(db)
        db = self.db_activation_3(db)
        
        
        # Compute Only(no Trainable params from here)
        #########################################
        ub = tf.pad(db, [[0,0],[1,0]], constant_values=0) # batch_size x n_data_points
        ub = tf.cumsum(ub, axis=1)
        ub = tf.minimum(ub, 1)
        ub = tf.expand_dims(ub, axis=-1) # 1 x n_data_points x 1
        
        # Bezier layer
        # Compute values of basis functions at data points
        num_control_points = self.bezier_degree + 1
        lbs = tf.tile(ub, [1, 1, num_control_points]) # batch_size x n_data_points x n_control_points
        pw1 = tf.range(0, num_control_points, dtype=tf.float32)
        pw1 = tf.reshape(pw1, [1, 1, -1]) # 1 x 1 x n_control_points
        pw2 = tf.reverse(pw1, axis=[-1])
        lbs = tf.add(tf.multiply(pw1, tf.math.log(lbs+self.EPSILON)), tf.multiply(pw2, tf.math.log(1-lbs+self.EPSILON))) # batch_size x n_data_points x n_control_points
        lc = tf.add(tf.math.lgamma(pw1+1), tf.math.lgamma(pw2+1))
        lc = tf.subtract(tf.math.lgamma(tf.cast(num_control_points, dtype=tf.float32)), lc) # 1 x 1 x n_control_points
        lbs = tf.add(lbs, lc) # batch_size x n_data_points x n_control_points
        bs = tf.exp(lbs)
        
        # Compute data points
        cp_w = tf.multiply(cp, w)
        dp = tf.matmul(bs, cp_w) # batch_size x n_data_points x 2
        bs_w = tf.matmul(bs, w) # batch_size x n_data_points x 1
        dp = tf.math.divide(dp, bs_w) # batch_size x n_data_points x 2
        dp = tf.expand_dims(dp, axis=-1) # batch_size x n_data_points x 2 x 1

        return dp, cp, w, ub, db

class Surrogate_Lift2Drag(keras.Model):
    def __init__(self):
        super(Surrogate_Lift2Drag, self).__init__()
        
        self.surrogate_depth = 16
        self.surrogate_kernel_size = (4,2)
        self.surrogate_residual_list = [2, 2, 2, 2]
        self.surrogate_weight_init = keras.initializers.VarianceScaling()
        self.surrogate_weight_regularizer = keras.regularizers.l2(0.0001)
        
        self.layer_1 = keras.layers.Conv2D(self.surrogate_depth * 1, self.surrogate_kernel_size, strides=1, 
                                                     padding='same', kernel_initializer=self.surrogate_weight_init, 
                                                     kernel_regularizer=self.surrogate_weight_regularizer)
        self.res_layer_set = []
        for i in range(self.surrogate_residual_list[0]):
            self.res_layer_set.append(Residual_Block(channels=self.surrogate_depth * 1, kernel_size=self.surrogate_kernel_size, 
                                                    downsample=False, kernel_initializer=self.surrogate_weight_init, 
                                                    kernel_regularizer=self.surrogate_weight_regularizer))
        
        self.res_layer_set.append(Residual_Block(channels=self.surrogate_depth * 2, kernel_size=self.surrogate_kernel_size, 
                                                    downsample=True, kernel_initializer=self.surrogate_weight_init, 
                                                    kernel_regularizer=self.surrogate_weight_regularizer))
        for i in range(1, self.surrogate_residual_list[1]):
            self.res_layer_set.append(Residual_Block(channels=self.surrogate_depth * 2, kernel_size=self.surrogate_kernel_size, 
                                                    downsample=False, kernel_initializer=self.surrogate_weight_init, 
                                                    kernel_regularizer=self.surrogate_weight_regularizer))
            
        self.res_layer_set.append(Residual_Block(channels=self.surrogate_depth * 4, kernel_size=self.surrogate_kernel_size, 
                                                    downsample=True, kernel_initializer=self.surrogate_weight_init, 
                                                    kernel_regularizer=self.surrogate_weight_regularizer))
        for i in range(1, self.surrogate_residual_list[2]):
            self.res_layer_set.append(Residual_Block(channels=self.surrogate_depth * 4, kernel_size=self.surrogate_kernel_size, 
                                                    downsample=False, kernel_initializer=self.surrogate_weight_init, 
                                                    kernel_regularizer=self.surrogate_weight_regularizer))
        
        self.res_layer_set.append(Residual_Block(channels=self.surrogate_depth * 8, kernel_size=self.surrogate_kernel_size, 
                                                    downsample=True, kernel_initializer=self.surrogate_weight_init, 
                                                    kernel_regularizer=self.surrogate_weight_regularizer))
        for i in range(1, self.surrogate_residual_list[3]):
            self.res_layer_set.append(Residual_Block(channels=self.surrogate_depth * 8, kernel_size=self.surrogate_kernel_size, 
                                                    downsample=False, kernel_initializer=self.surrogate_weight_init, 
                                                    kernel_regularizer=self.surrogate_weight_regularizer))
        
        self.batchnorm_1 = keras.layers.BatchNormalization(momentum=0.9)
        self.LRelu_1 = keras.layers.LeakyReLU(alpha=0.2)
        self.flatten = keras.layers.Flatten()
        self.layer_2 = keras.layers.Dense(128, kernel_initializer=self.surrogate_weight_init, 
                                          kernel_regularizer=self.surrogate_weight_regularizer)
        self.batchnorm_2 = keras.layers.BatchNormalization(momentum=0.9)
        self.LRelu_2 = keras.layers.LeakyReLU(alpha=0.2)
        self.layer_3 = keras.layers.Dense(1, kernel_initializer=self.surrogate_weight_init, 
                                          kernel_regularizer=self.surrogate_weight_regularizer)
        self.sigmoid = keras.layers.Activation(keras.activations.sigmoid)
        
    def call(self, inputs, training = True):
        x = inputs
        x = self.layer_1(x)
        
        for i in range(len(self.res_layer_set)):
            x = self.res_layer_set[i](x, training = training)
            
        x = self.batchnorm_1(x, training = training)
        x = self.LRelu_1(x)
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = self.flatten(x)
        x = self.layer_2(x)
        x = self.batchnorm_2(x, training = training)
        x = self.LRelu_2(x)
        x = self.layer_3(x)
        x = self.sigmoid(x)
        
        return x
        
class Bezier_PaDGAN(object):
    def __init__(self, latent_dim=5, noise_dim=10, n_points=192, bezier_degree=31, bounds=(0., 1.), lambda0=2.0, lambda1=0.2):

        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.bounds = bounds
        
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        
        self.EPSILON = 1e-7
        self.EPSILON_gen = 1e-7
        
        #Generator Setup:
        self.generator = Bezier_Generator(self.latent_dim, self.noise_dim, bezier_degree, (n_points, 2, 1), self.EPSILON_gen)
        
        
        #Discriminator Setup
        self.discriminator = Bezier_Discriminator(self.latent_dim, self.EPSILON)
        
        #Surrogate Model Setup
        self.surrogate = Surrogate_Lift2Drag()
    
    @tf.function 
    def train_surrogate_step_graph(self, X_batch, Y_batch, X_test, Y_test, optimizer, loss_fn):
        
        with tf.GradientTape() as tape:
            Y_pred = self.surrogate(X_batch)
            loss = loss_fn(Y_batch,Y_pred)
        variables = self.surrogate.trainable_weights
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        Y_pred_test = self.surrogate(X_test)
        loss_test = loss_fn(Y_test,Y_pred_test)
        return loss, loss_test
    
    def train_surrogate(self, X_train, Y_train, X_test, Y_test, steps = 10000, batch_size = 256, lr = 1e-4):
        steps_range = trange(steps, desc='Surrogate Model', leave=True, ascii ="         =")
        optimizer = keras.optimizers.Adam(learning_rate = lr, beta_1 = 0.5)
        loss_fn = keras.losses.MeanAbsoluteError()
        for step in steps_range:
            ind = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
            X_batch = X_train[ind]
            Y_batch = Y_train[ind]
            loss, loss_test = self.train_surrogate_step_graph(X_batch, Y_batch, X_test, Y_test, optimizer, loss_fn)
            steps_range.set_description("loss = %f , test_loss = %f"%(loss, loss_test))
    
    @tf.function
    def compute_diversity_loss(self, x, y):  
        flatten_fn = keras.layers.Flatten()    
        x = flatten_fn(x)
        y = tf.squeeze(y)
        
        r = tf.reduce_sum(tf.math.square(x), axis=1, keepdims=True)
        D = r - 2*tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
        S = tf.exp(-0.5*tf.math.square(D)) # similarity matrix (rbf)
        
        if self.lambda0 == 'naive':
            
            eig_val, _ = tf.linalg.eigh(S)
            loss = -tf.reduce_mean(tf.math.log(tf.maximum(eig_val, self.EPSILON)))-10*tf.reduce_mean(y)
            
            Q = None
            L = None
            
        else:
            
            Q = tf.tensordot(tf.expand_dims(y, 1), tf.expand_dims(y, 0), 1) # quality matrix
            if self.lambda0 == 0.:
                L = S
            else:
                L = S * tf.math.pow(Q, self.lambda0)
            
            eig_val, _ = tf.linalg.eigh(L)
            loss = -tf.reduce_mean(tf.math.log(tf.maximum(eig_val, self.EPSILON)))
        
        return loss, D, S, Q, L
    
    @tf.function
    def disc_loss_real(self, X_real):
        binary_cross_entropy_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        #compute model D predictions for real sample
        d_real, _ = self.discriminator(X_real, training = True)
        
        #discriminator loss calculations
        d_loss_real = binary_cross_entropy_loss_fn(tf.ones_like(d_real),d_real)
        
        return d_loss_real
    
    @tf.function
    def disc_loss_fake(self, x_fake_train, c):
        binary_cross_entropy_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        d_fake, q_fake_train = self.discriminator(x_fake_train, training = True)
        #discriminator loss calculations
        d_loss_fake = binary_cross_entropy_loss_fn(tf.zeros_like(d_fake),d_fake)
        # Gaussian loss for Q
        q_mean = q_fake_train[:, 0, :]
        q_logstd = q_fake_train[:, 1, :]
        epsilon = (c - q_mean) / (tf.math.exp(q_logstd) + self.EPSILON)
        q_loss = q_logstd + 0.5 * tf.math.square(epsilon)
        q_loss = tf.reduce_mean(q_loss)
        #discriminator loss
        total_d_loss = q_loss + d_loss_fake
        
        return total_d_loss, d_loss_fake, q_loss
    
    @tf.function
    def gen_loss(self, batch_size, lambda1):
        
        binary_cross_entropy_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        c = tf.random.uniform((batch_size,self.latent_dim), minval = self.bounds[0], maxval = self.bounds[1], dtype=tf.float32)
        z = tf.random.normal((batch_size,self.noise_dim), stddev = 0.5, dtype=tf.float32)

        x_fake_train, cp_train, w_train, ub_train, db_train = self.generator(c, z, training = True)
        d_fake, q_fake_train = self.discriminator(x_fake_train, training = True)
        y = self.surrogate(x_fake_train, training = False)
        mean_y = tf.reduce_mean(y)

        #generator loss calculations
        g_loss = binary_cross_entropy_loss_fn(tf.ones_like(d_fake),d_fake)
        dpp_loss, D, S, Q, L = self.compute_diversity_loss(x_fake_train, y * d_fake)
        g_dpp_loss = g_loss + lambda1 * dpp_loss
        #regularization for w, cp, a, and b
        r_w_loss = tf.reduce_mean(w_train[:,1:-1], axis=[1,2])
        cp_dist = tf.norm(cp_train[:,1:] - cp_train[:,:-1], axis=-1)
        r_cp_loss = tf.reduce_mean(cp_dist, axis=-1)
        r_cp_loss1 = tf.reduce_max(cp_dist, axis=-1)
        ends = cp_train[:,0] - cp_train[:,-1]
        r_ends_loss = tf.norm(ends, axis=-1) + tf.maximum(0.0, -10*ends[:,1])
        r_db_loss = tf.reduce_mean(db_train * tf.math.log(db_train), axis=-1)
        r_loss = r_w_loss + r_cp_loss + 0 * r_cp_loss1 + r_ends_loss + 0 * r_db_loss
        r_loss = tf.reduce_mean(r_loss)
        #gaussian loss for Q
        q_mean = q_fake_train[:, 0, :]
        q_logstd = q_fake_train[:, 1, :]
        epsilon = (c - q_mean) / (tf.math.exp(q_logstd) + self.EPSILON)
        q_loss = q_logstd + 0.5 * tf.math.square(epsilon)
        q_loss = tf.reduce_mean(q_loss)
        #generator loss
        total_g_loss = g_dpp_loss + 10 * r_loss + q_loss
        
        return total_g_loss, g_loss, q_loss, dpp_loss, mean_y, r_loss
        
    @tf.function
    def train_GAN_step_graph(self, X_real, d_optimizer, g_optimizer, batch_size, lambda1):
    
        #Discriminator Training
        #########################################################################################################
        d_optimizer.minimize(lambda: self.disc_loss_real(X_real), lambda: self.discriminator.trainable_weights)
        
        #########################################################################################################
        #generate fake samples
        c = tf.random.uniform((batch_size,self.latent_dim), minval = self.bounds[0], maxval = self.bounds[1], dtype=tf.float32)
        z = tf.random.normal((batch_size,self.noise_dim), stddev = 0.5, dtype=tf.float32)
        x_fake_train= self.generator(c, z)[0]
        
        d_optimizer.minimize(lambda: self.disc_loss_fake(x_fake_train, c)[0], lambda: self.discriminator.trainable_weights)
        #########################################################################################################

        #Generator Training
        #########################################################################################################
        g_optimizer.minimize(lambda: self.gen_loss(batch_size, lambda1)[0], self.generator.trainable_weights)
        
        
    def train_GAN(self, X_train, steps=10000, batch_size=32, disc_lr=1e-4, gen_lr=1e-4):
        
        disc_lr = tf.keras.optimizers.schedules.ExponentialDecay(disc_lr, 2000, 0.8, staircase=True)
        gen_lr = tf.keras.optimizers.schedules.ExponentialDecay(gen_lr, 1000, 0.8, staircase=True)
        
        g_optimizer = keras.optimizers.Adam(learning_rate = gen_lr, beta_1=0.5)
        d_optimizer = keras.optimizers.Adam(learning_rate = disc_lr, beta_1=0.5)
        
        steps_range = trange(steps, position=0, desc='GAN Training', leave=True, ascii ="         =")
        for step in steps_range:
            p = tf.cast(5.0, tf.float32)
            lambda1 = self.lambda1 * tf.cast(step/(steps-1), tf.float32) ** p
            ind = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
            X_real = X_train[ind]
            self.train_GAN_step_graph(X_real, d_optimizer, g_optimizer, batch_size, lambda1)
            
            if step % 10 == 0:
                
                d_loss_real = self.disc_loss_real(X_real)

                c = tf.random.uniform((batch_size,self.latent_dim), minval = self.bounds[0], maxval = self.bounds[1], dtype=tf.float32)
                z = tf.random.normal((batch_size,self.noise_dim), stddev = 0.5, dtype=tf.float32)
                x_fake_train= self.generator(c, z)[0]
                total_d_loss, d_loss_fake, q_d_loss = self.disc_loss_fake(x_fake_train, c)
                
                total_g_loss, g_loss, q_loss, dpp_loss, mean_y, r_loss = self.gen_loss(batch_size, lambda1)

                steps_range.set_postfix_str(
                    "[D] R = %+.7f, F = %+.7f, q = %+.7f, lr = %+.7f [G] F = %+.7f, q = %.7f, dpp = %+.7f, reg = %+.7f, y = %+.7f, lambda1 = %+.7f" % (d_loss_real,d_loss_fake,q_d_loss, gen_lr(step), g_loss, q_loss, dpp_loss, r_loss, mean_y, lambda1))
            
    def save_model(self, directory = '.', name = "PadGANBezier", model = "surrogate generator discriminator"):
        if not os.path.exists(directory + "//surrogate"):
            os.mkdir(directory + "//surrogate")
            
        if not os.path.exists(directory + "//generator"):
            os.mkdir(directory + "//generator")
            
        if not os.path.exists(directory + "//discriminitor"):
            os.mkdir(directory + "//discriminitor")
        
        if "surrogate" in model:
            self.surrogate.save_weights(directory + "//surrogate//" + name)
        if "generator" in model:
            self.generator.save_weights(directory + "//generator//" + name)
        if "discriminator" in model:
            self.discriminator.save_weights(directory + "//discriminitor//" + name)
    
    def load_model(self, directory = '.', name = "PadGANBezier", model = "surrogate generator discriminator"):
        if "surrogate" in model:
            self.surrogate.load_weights(directory + "\\surrogate\\" + name)
        if "generator" in model:
            self.generator.load_weights(directory + "\\generator\\" + name)
        if "discriminator" in model:
            self.discriminator.load_weights(directory + "\\discriminitor\\" + name)
    
    def generate_sample(self, n):
        c = tf.random.uniform((n,self.latent_dim), minval = self.bounds[0], maxval = self.bounds[1], dtype=tf.float32)
        z = tf.random.normal((n,self.noise_dim), stddev = 0.5, dtype=tf.float32)
        x_generated = self.generator(c, z)[0]
        return x_generated