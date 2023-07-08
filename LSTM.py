
#%%
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import keras.backend as K
import tensorflow as tf
from numpy import newaxis as na

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
        timesteps = input_data.shape[1]
        print("timesteps", timesteps)
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
        for t in range(timesteps):
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

    def lstm_lrp_rudder(self, input_data, rel_prev):
        """_summary_

        Args:
            input_data (array): (batch_size, timesteps, dimensions)
            rel_prev (array): relevance is initialised by this value.

        Returns:
            array: relevance score for this lstm layer
            
        """
        
        
        # source: https://arxiv.org/pdf/1806.07857.pdf
        
        # compute all activations (gates and signals for lstm layer)
        self.get_lstm_states(input_data)
        
        
        if len(rel_prev.shape) == 2:
            rel_prev = rel_prev[-1, :]
        
        # initialise relevance
        RyT = rel_prev # (M, )  M...output dim of current layer
        

        
        # Get the LSTM weights and biases
        W = self.get_weights()[0]  # LSTM weights (D, 4M) - 4 because of 4 gates
        w_i, w_f, w_c, w_o = tf.split(W, num_or_size_splits=4, axis=1) # (D,M), (D,M), (D,M), (D,M)
        
        # Get the LSTM weights and biases
        b = self.get_weights()[2]  # Biases (4M,)
        b_i, b_f, b_c, b_o = tf.split(b, num_or_size_splits=4, axis=0) # (M,), (M,), (M,), (M,)

        print("--asdfasdfasdf----")
    
        # Extract the last cell state
        cT = np.array(self.cell_states)[-1, 0, :] # (timesteps, batch_size, dimensions) -> (dimensions, )

        # Collect relevance scores
        relevance = []

        for t in reversed(range(timesteps)): 
            
            
            # rules according to Rudder
            zt = np.array(self.cell_input_signal)[t, 0, :] # (timesteps, batch_size, dimensions) -> (dimensions,)
            it = np.array(self.input_gate_activation)[t, 0, :] # (timesteps, batch_size, dimensions) -> (dimensions,)
            
            
            Rzt =  (zt * it) * RyT / cT
            
            print("ööööööööööööööööööööööööööööö")
            print("zt.shape",zt.shape)
            print("it.shape", it.shape)
            print("cT.shape", cT.shape)
            print("RyT.shape", RyT.shape)
            print("Rzt.shape", Rzt.shape)
            
            
            # using linear rule
            relevance.append(lrp_linear(np.array(w_c), np.array(b_c), input_data[0, t, :], zt, Rzt, 3))
        
        
        print("LRP LSTM - DONE")
        return np.array(relevance)



# Define your model class
class LSTMModel(tf.keras.Model):
    
    def __init__(self):
        super(LSTMModel, self).__init__()
        
        lstm_units = 50
        input_dim = 3
        timesteps = 5
        
        # set return_state = True to return the last state in addition to the output. Default: False. 
        self.lstm1 = CustomLSTM(units=lstm_units, input_shape=(timesteps, input_dim),
                                          return_sequences=True, return_state=True)
        self.lstm2 = CustomLSTM(units=25, return_sequences=False, return_state=True)
        self.dense1 = tf.keras.layers.Dense(8)
        self.dense2 = tf.keras.layers.Dense(1)  # Output layer
        
    def call(self, inputs):
        
        
        # LSTM
        lstm1_output, hidden_state1, cell_state1 = self.lstm1(inputs)
        lstm2_output, hidden_state2, cell_state2 = self.lstm2(lstm1_output)
        
        # Dense
        dense1_output = self.dense1(lstm2_output)
    
        
        # Apply dense2 layer for final output
        output = self.dense2(dense1_output)
        
        
        # append all outputs to giant list
        self.activations = [{"output"      : lstm1_output.numpy(), 
                             "hidden_state": hidden_state1.numpy(),
                             "cell_state"  : cell_state1.numpy()},
                            {"output"      : lstm2_output.numpy(), 
                             "hidden_state": hidden_state2.numpy(),
                             "cell_state"  : cell_state2.numpy()},
                            {"output"      : dense1_output.numpy()},
                            {"output"      : output.numpy()}]
        
        # final output
        return output
    

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
model.activations[-4]["output"].shape


# outputs: (1, 5, 50) -> (1, 25) -> (1, 8) -> (1, 1)



#%%

def backpropagate_relevance(model, input_data):
    
    # feedforward
    out = model(input_data)
    
    # get prediction and initialise relevance scores
    Rj = out.numpy()[0]
    
    print("Rj:", Rj)
    
    # Iterate through each layer in reverse order
    for i in reversed(range(len(model.layers))):
        
        # get the current layer and the lower layer
        current_layer = model.layers[i]
        lower_layer   = model.layers[i-1]
        

        
        # print info
        if i >0:
            print("Backpropagating Rel for Layer", current_layer, "to", lower_layer)
        
        # linear to linear
        if isinstance(current_layer, tf.keras.layers.Dense): 
            
             # get the number of output-nodes in the lower layer
            num_nodes_in_lower_layer = model.activations[i-1]["output"].\
                shape[1] #  (batch_size, timesteps, D) -> (D, )
        
            
            w = current_layer.get_weights()[0] # get the weights for the last layer
            b = current_layer.get_weights()[1] # get the biases for the last layer
            
            

            # if last layer -> inititalise zj with the final prediction
            if (i == len(model.layers) - 1): 
                zj = Rj 
            else:
                zj = model.activations[i]["output"][0,:] # shape (M, )
            
            # compute the activations in the lower layer
            zi = model.activations[i-1]["output"][0,:] # shape (D, )
            
            # compute by relevance of layer by linear rule
            Rj = lrp_linear(w, b, zi, zj, Rj, num_nodes_in_lower_layer)
            
        elif isinstance(current_layer, tf.keras.layers.LSTM):
            
            #print(model.activations[i-1]["output"].shape)
            print(Rj.shape)
            
            if i == 0:
                Rj = current_layer.lstm_lrp_rudder(input_data, Rj)
            else:
            
                input_tmp = model.activations[i-1]["output"] 

                Rj = current_layer.lstm_lrp_rudder(input_tmp, Rj)

            
            
    return Rj


backpropagate_relevance(model, input_data)
# %%
