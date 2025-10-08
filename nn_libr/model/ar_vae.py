import numpy as np
import tensorflow as tf
import os
import sys

sys.path.append('../../') 

from tensorflow import keras
from nn_libr.layers.SamplingLayer import SamplingLayer



def compute_attribute_loss(latent_var, attribute, delta=1.0):
    # Compute input distance matrix
    lv_flat = tf.expand_dims(tf.reshape(latent_var, [-1]), axis=-1)
    L_lv = tf.tile(lv_flat, (1, lv_flat.shape[0]))
    D_lv = L_lv - tf.transpose(L_lv)
    
    # Compute target distance matrix
    attr_flat = tf.expand_dims(tf.reshape(attribute, [-1]), axis=-1)
    L_attr = tf.tile(attr_flat, (1, attr_flat.shape[0]))
    D_attr = L_attr - tf.transpose(L_attr)

    # Compute regularization loss
    input_tanh  = tf.math.tanh(D_lv * delta)
    target_sign = tf.math.sign(D_attr)
    mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
    loss = mae(input_tanh, target_sign)
    return loss



class AR_VAE(keras.Model):
    def __init__(self, encoder=None, decoder=None, alpha=1, beta=1, gamma=1, delta=1, **kwargs):
        '''
        AR_VAE instantiation with encoder, decoder and r_loss_factor
        args :
            encoder : Encoder model
            decoder : Decoder model
            alpha : Weight of the loss function: Recons_loss
            beta  : Weight of the loss function: Kl_loss
            gamma : Weight of the loss function: Reg_loss
            delta : Weight for the distance matrix in latent space
            r_loss_factor : Proportion of reconstruction loss for global loss (0.3)
        return:
            None
        '''
        super(AR_VAE, self).__init__(**kwargs)
        self.encoder      = encoder
        self.decoder      = decoder
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker  = tf.keras.metrics.Mean(name="kl_loss")
        self.reg_loss_tracker = tf.keras.metrics.Mean(name="reg_loss")
        print(f'\nVAE is ready : ==> \nAlpha = {self.alpha}; Beta = {self.beta}; Gamma = {self.gamma}; Delta = {self.delta}')
       

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.reg_loss_tracker,
        ]       
        
    def call(self, inputs):
        '''
        Model forward pass, when we use the model
        args:
            inputs : Model inputs
        return:
            output : Output of the model 
        '''
        z_mean, z_log_var, z = self.encoder(inputs)
        output               = self.decoder(z)
        return output
                
        
    def train_step(self, input):
        '''
        Implementation of the training update.
        Receive an input, compute loss, get gradient, update weights and return metrics.
        Here, the metrics are loss.
        args:
            inputs : Model inputs
        return:
            loss    : Total loss
            r_loss  : Reconstruction loss
            kl_loss : KL loss
        '''
                
        alpha = self.alpha        
        beta  = self.beta
        gamma = self.gamma
        delta = self.delta
        imgs, attribute = input
        nb_attr_dim = attribute.shape[-1]
        
        # ---- Forward pass
        #      Run the forward pass and record 
        #      operations on the GradientTape.
        #
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(imgs)
            reconstruction       = self.decoder(z)
         
            # ---- Compute loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mean_squared_error(imgs, reconstruction), axis=(1,2))
            ) * alpha

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = -tf.reduce_mean(kl_loss)*beta

            z_ar = z[:,:nb_attr_dim]
            reg_loss = compute_attribute_loss(z_ar, attribute, delta)*gamma

            total_loss = reconstruction_loss + kl_loss + reg_loss

        # ---- Retrieve gradients from gradient_tape
        #      and run one step of gradient descent
        #      to optimize trainable weights
        #
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.reg_loss_tracker.update_state(reg_loss)
        
        return {
            "loss":      total_loss,
            "rec_loss":  reconstruction_loss,
            "kl_loss":   kl_loss,
            "reg_loss":  reg_loss,
        }
    
    
    def predict(self,inputs):
        '''Our predict function...'''
        z_mean, z_var, z  = self.encoder.predict(inputs)
        outputs           = self.decoder.predict(z)
        return outputs

        
    def save(self,filename):
        '''Save model in 2 part'''
        filename, extension = os.path.splitext(filename)
        self.encoder.save(f'{filename}-encoder.h5')
        self.decoder.save(f'{filename}-decoder.h5')

    
    def reload(self,filename):
        '''Reload a 2 part saved model.'''
        filename, extension = os.path.splitext(filename)
        self.encoder = keras.models.load_model(f'{filename}-encoder.h5', custom_objects={'SamplingLayer': SamplingLayer})
        self.decoder = keras.models.load_model(f'{filename}-decoder.h5')
        print('\nVAE Reloaded.')
                
        