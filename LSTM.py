#%%
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import keras.backend as K
from numpy import newaxis as na
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# only display logging messages that correspond to errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import matplotlib.pyplot as plts
from tqdm import tqdm

import matplotlib.pyplot as plt

#%%


# Define custom 
class CustomLSTM(tf.keras.layers.LSTM):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        # Custom build logic if needed

    def call(self, inputs, **kwargs):
        # Custom call logic if needed
        return super().call(inputs, **kwargs)
    
    def get_lstm_states(self, input_data):
        """computes a forward pass through the lstm layer

        Args:
            input_data (array): (batch_size, timesteps, dimensions)
        """
    
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Get the LSTM weights and biases
        W = self.get_weights()[0]  # LSTM weights (D, 4M) - 4 because of 4 gates
        w_i, w_f, w_c, w_o = tf.split(W, num_or_size_splits=4, axis=1)

        H = self.get_weights()[1]  # Recurrent weights (M, 4M)
        H_i, H_f, H_c, H_o = tf.split(H, num_or_size_splits=4, axis=1)

        b = self.get_weights()[2]  # Biases (4M,)
        b_i, b_f, b_c, b_o = tf.split(b, num_or_size_splits=4, axis=0)

        # Get the number of timesteps and hidden units
        self.timesteps = input_data.shape[1]
        
        hidden_units = self.units

        # Initialize lists to store the hidden states and cell states
        self.hidden_states = []
        self.cell_states = []

        # Initialise list to store activations of gates (sigmoid) and signals (tanh)
        self.input_gate_activation = []
        self.forget_gate_activation = []
        self.output_gate_activation = []
        self.cell_input_signal = []
        self.cell_state_signal = []

        # Initialize the initial hidden state and cell state as zeros
        h_prev = np.zeros((1, hidden_units))
        c_prev = np.zeros((1, hidden_units))

        # Perform forward pass for each timestep
        for t in range(self.timesteps):
            # Get the input at the current timestep
            
            x = input_data[:, t, :]
            
            # Compute activations for the forget gate
            forget_gate_activation = np.dot(x, w_f) + np.dot(h_prev, H_f) + b_f
            forget_gate_activation = sigmoid(forget_gate_activation)

            # Compute activations for the output gate
            output_gate_activation = np.dot(x, w_o) + np.dot(h_prev, H_o) + b_o
            output_gate_activation = sigmoid(output_gate_activation)

            # Compute activations for the input gate
            input_gate_activation = np.dot(x, w_i) + np.dot(h_prev, H_i) + b_i
            input_gate_activation = sigmoid(input_gate_activation)

            # Compute signals using the tanh activation function

            # Compute signal for the cell input (part of the input gate)
            cell_input_signal = np.dot(x, w_c) + np.dot(h_prev, H_c) + b_c
            cell_input_signal = np.tanh(cell_input_signal)


            # Update the cell state
            c_new = forget_gate_activation * c_prev + input_gate_activation * cell_input_signal

            # Compute the cell state signal
            cell_state_signal = np.tanh(c_new)

            # Update hidden state 
            h_new = cell_state_signal * output_gate_activation
            
            # Append the current states to the lists
            self.hidden_states.append(np.array(h_new)) # (time_steps, batch_size, dimensions )
            self.cell_states.append(np.array(c_new)) # (time_steps, batch_size, dimensions )
            
            # Append activations
            
            self.input_gate_activation.append( np.array(input_gate_activation) )  # (time_steps, batch_size, dimensions )
            self.forget_gate_activation.append( np.array(forget_gate_activation) ) # (time_steps, batch_size, dimensions )
            self.output_gate_activation.append( np.array(output_gate_activation) ) # (time_steps, batch_size, dimensions )
            self.cell_input_signal.append( np.array(cell_input_signal) ) # (time_steps, batch_size, dimensions )
            self.cell_state_signal.append( np.array(cell_state_signal) ) # (time_steps, batch_size, dimensions )

    def lstm_lrp_rudder(self, input_data, rel_prev, aggregate=True):
        """_summary_

        Args:
            input_data (array): (batch_size, timesteps, dimensions)
            rel_prev (array): relevance is initialised by this value.

        Returns:
            array: relevance score for this lstm layer
        
        
        source: https://arxiv.org/pdf/1806.07857.pdf
        """
        
        # compute all activations (gates and signals for lstm layer)
        self.get_lstm_states(input_data)
        
        # the output of an LSTM layer with return_sequence=True must be handled -> 
        # if return_sequence=True then rel_prev.shape = (timestpes, M)
        # we can either take option 1) the last relevance scores in the sequence
        #                    option 2) aggreagte the relevance scores of the sequence
        
        
        #print("lstm_lrp_rudder - aggregate: ",aggregate)
        
        if len(rel_prev.shape) == 2 and aggregate:
            #print("DO AGGREGATE")
            rel_prev = np.mean(rel_prev, axis=0)
            
        elif len(rel_prev.shape) == 2 and not aggregate:
            #print("DO NOT AGGREGATE")
            rel_prev = rel_prev[-1, :]
            
        
        
        # initialise relevance
        RyT = rel_prev # (M, )  M...output dim of current layer
        
        # Get the LSTM weights and biases
        W = self.get_weights()[0]  # LSTM weights (D, 4M) - 4 because of 4 gates
        w_i, w_f, w_c, w_o = tf.split(W, num_or_size_splits=4, axis=1) # (D,M), (D,M), (D,M), (D,M)
        
        # Get the LSTM weights and biases
        b = self.get_weights()[2]  # Biases (4M,)
        b_i, b_f, b_c, b_o = tf.split(b, num_or_size_splits=4, axis=0) # (M,), (M,), (M,), (M,)
    
        # Extract the last cell state
        cT = np.array(self.cell_states)[-1, 0, :] # (timesteps, batch_size, dimensions) -> (dimensions, )

        
        # Collect relevance scores
        relevance = []

        for t in reversed(range(self.timesteps)): 
            
            
            # rules according to Rudder
            zt = np.array(self.cell_input_signal)[t, 0, :] # (timesteps, batch_size, dimensions) -> (dimensions,)
            it = np.array(self.input_gate_activation)[t, 0, :] # (timesteps, batch_size, dimensions) -> (dimensions,)
            
            
            Rzt =  (zt * it) * RyT / cT
            
            # for debugging
            #print("zt.shape",zt.shape)
            #print("it.shape", it.shape)
            #print("cT.shape", cT.shape)
            #print("RyT.shape", RyT.shape)
            #print("Rzt.shape", Rzt.shape)
            
            
            # using linear rule
            relevance.append(lrp_linear(np.array(w_c), np.array(b_c), input_data[0, t, :], zt, Rzt, 3))
        
        
        #print("LRP LSTM - DONE")
        return np.array(relevance)



# Define your model class
class LSTMModel(tf.keras.Model):
    
    def __init__(self, lstm_units, input_dim, timesteps, l2_lambda=0.02): # calls the constructor of the child class (LSTMModel)
        super(LSTMModel, self).__init__() # calls constructor of parent class: tf.keras.Model
        
        # Set return_state = True to return the last state in addition to the output. Default: False. 
        self.lstm1 = CustomLSTM(
            units=lstm_units,
            input_shape=(timesteps, input_dim),
            return_sequences=True,
            return_state=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)  # L2 regularization for the first LSTM layer
        )
        self.lstm2 = CustomLSTM(
            units=32,
            return_sequences=False,
            return_state=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)  # L2 regularization for the second LSTM layer
        )
        self.dense1 = tf.keras.layers.Dense(
            16,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)  # L2 regularization for the first dense layer
        )
        self.dense2 = tf.keras.layers.Dense(
            1,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)  # L2 regularization for the output layer
        )
        
    def call(self, inputs):
        
        
        # LSTM
        lstm1_output, hidden_state1, cell_state1 = self.lstm1(inputs)
        lstm2_output, hidden_state2, cell_state2 = self.lstm2(lstm1_output)
        
        # Dense
        dense1_output = self.dense1(lstm2_output)
    
        
        # Apply dense2 layer for final output
        output = self.dense2(dense1_output)
        
        
        # append all activation of the forward pass to dictionary
        self.activations = [{"output": lstm1_output,
                        "hidden_state": hidden_state1,
                        "cell_state": cell_state1},
                    {"output": lstm2_output,
                        "hidden_state": hidden_state2,
                        "cell_state": cell_state2},
                    {"output": dense1_output},
                    {"output": output}]
        
        # final output
        return output
    
    
    def get_activations(self):
        """Return the activations after the last forward pass."""
        
        
        new_list = []
        
        for dic in self.activations:
            
            new_dic = {}
            
            for key, item in dic:
                new_dic["key"] = item.numpy()
                
            new_list.append(new_dic)
        
        return new_list
    
    def backpropagate_relevance(self, input_data, aggregate):
    
        # feedforward
        out = self(input_data)
        
        # get prediction and initialise relevance scores
        Rj = out.numpy()[0]
        
        # Iterate through each layer in reverse order
        for i in reversed(range(len(self.layers))):
            
            # get the current layer and the lower layer
            current_layer = self.layers[i]
            lower_layer   = self.layers[i-1]
            
            # print info
            # if i >0:
            #     print("Backpropagating Rel for Layer", current_layer, "to", lower_layer)
            
            # linear to linear
            if isinstance(current_layer, tf.keras.layers.Dense): 
                
                # get the number of output-nodes in the lower layer
                num_nodes_in_lower_layer = self.activations[i-1]["output"].numpy().shape[1] #  (batch_size, timesteps, D) -> (D, )
            
                
                w = current_layer.get_weights()[0] # get the weights for the last layer
                b = current_layer.get_weights()[1] # get the biases for the last layer
                
                

                # if last layer -> inititalise zj with the final prediction
                if (i == len(self.layers) - 1): 
                    zj = Rj 
                else:
                    zj = self.activations[i]["output"].numpy()[0,:] # shape (M, )
                
                # compute the activations in the lower layer
                zi = self.activations[i-1]["output"].numpy()[0,:] # shape (D, )
                
                # compute by relevance of layer by linear rule
                Rj = lrp_linear(w, b, zi, zj, Rj, num_nodes_in_lower_layer)
                
            elif isinstance(current_layer, tf.keras.layers.LSTM):
                
                #print(self.activations[i-1]["output"].shape)
                
                if i == 0:
                    Rj = current_layer.lstm_lrp_rudder(input_data, Rj, aggregate)
                else:
                
                    input_tmp = self.activations[i-1]["output"].numpy()

                    # print("aggregate:", aggregate)
                    Rj = current_layer.lstm_lrp_rudder(input_tmp, Rj, aggregate)

            # print("Rj.shape", Rj.shape)
                
        return Rj
    
    


def rolling_fit(X_train, y_train, big_window_size=60, validation_size=1/60, small_window_size=5, do_plot=True):

    # Define the window size to train the model on
    big_window_size = 60

    # Define the size of the validation set for the rolling window
    validation_size = 1 / 60

    # Calculate the number of rolling windows and the size of the validation set
    num_windows = len(X_train) - big_window_size 
    validation_set_size = int(num_windows * validation_size)

    # collect the predictions in a list
    predictions = []

    # collect relevance scores in a list
    relevance = []

    for i in tqdm(range(num_windows)):
        start_index = i
        end_index = i + big_window_size

        X_train_window = X_train[start_index:end_index]
        y_train_window = y_train[start_index:end_index]
        
        y_train_window = y_train_window.reshape(-1, 1)

        # Create a validation set from a portion of the rolling window
        X_val = X_train[end_index : end_index + validation_set_size]
        y_val = y_train[end_index : end_index + validation_set_size]
        
        y_val = y_val.reshape(-1, 1)
        
        # Define a new model at each iteration
        model = LSTMModel(lstm_units=64, input_dim=X_train.shape[2], 
                          timesteps=small_window_size, l2_lambda=0.01)

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Define the EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=0)

        # Debugging:
        # print("X_train_window.shape", X_train_window.shape)
        # print("y_train_window.shape", y_train_window.shape)
        # print("y_val.shape", y_val.shape)

        # Train the model on the current rolling window
        model.fit(X_train_window,
                y_train_window,
                validation_data=(X_val, y_val),
                epochs=10,
                batch_size=32,
                verbose=False,
                callbacks=[early_stopping])

        # Predict the next timestep
        X_next = X_train[end_index:end_index + 1]  # Get the last observation in the validation set

        next_pred = model.predict(X_next, verbose=False)  # Predict the next timestep based on X_next
        predictions.append(next_pred)

        rel = model.backpropagate_relevance(X_next, False)
        relevance.append(rel)


    predictions =  np.array(predictions)[:, 0, 0]
    
    if do_plot:
        plt.plot(y_train[big_window_size:], label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.xlabel('Timestep')
        plt.ylabel('Response')
        plt.legend()
        plt.show()

    return predictions, relevance

def lrp_linear(w, b, z_i, z_j, Rj, nlower, eps=1e-4, delta=0.0):
    """
    LRP for a linear layer with input (previous layer) dim D and output (next layer) dim M.
    Args:
    
    w:   weights from layer i (lower) to j (higher) - array of shape (D, M)
    b:   biases  from layer i (lower) to j (higher) - array of shape (M, )
    z_i: linear activation of node i in lower layer - array of shape (D, )
    z_j: linear activation of node j in upper layer - array of shape (M, )
    Rj:  relevance score of node j from upper layer - array of shape (M, )
    nlower: the number of nodes in the lower layer  - this will be 
    eps: correction error for stabilisation i.e. to avoid cases like 0/0
    delta: set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
    
    Returns:
    
    Ri: relevance score for lower layer node - array of shape (D, )
    
    '''
    @author: Leila Arras
    @maintainer: Leila Arras
    @date: 21.06.2017
    @version: 1.0+
    @copyright: Copyright (c) 2017, Leila Arras, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
    @license: see LICENSE file in repository root
    '''
    """
    
    sign_out = np.where(z_j[na,:]>=0, 1., -1.) # shape (1, M)
    
    # define the numerator
    numer = (w * z_i[:,na]) + ( delta * (b[na,:] + eps * sign_out ) / nlower ) # shape (D, M)
    
    # Note: here we multiply the bias_factor with both the bias b and the stabilizer eps since in fact
    # using the term (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units in the numerator is only useful for sanity check
    # (in the initial paper version we were using (bias_factor*b[na,:]*1. + eps*sign_out*1.) / bias_nb_units instead)
    
    denom = z_j[na,:] + (eps*sign_out*1.)   # shape (1, M)
    
    message = (numer/denom) * Rj[na,:]       # shape (D, M)
    
    Rin = message.sum(axis=1)              # shape (D,)

    return Rin






if __name__ == "__main__":
    
    # test setup ------------------------------------------------------------------------------------
    # Create an instance of your custom model
    model = LSTMModel()

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    batch_size = 1  # Number of samples in a batch
    timesteps = 5  # Number of time steps
    input_dim = 3  # Dimensionality of each input

    input_data = np.random.rand(batch_size, timesteps, input_dim) # sample input
    # ---------------------------------------------------------------------------------------------

    out = model(input_data)

    #if model.layers[1].return_sequences: #<- built this into the backpropagation algo

    #model.activations[0]["hidden_state"]

    for ac in model.activations:
        print(ac["output"].shape)


    # outputs: (1, 5, 50) -> (1, 25) -> (1, 8) -> (1, 1)

    print(model.backpropagate_relevance(input_data, False))

