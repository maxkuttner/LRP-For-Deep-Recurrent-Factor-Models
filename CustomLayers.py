# Dependencies:
import numpy as np
import tensorflow as tf
from LRPMethods import lrp_linear


# Define custom LSTM layer
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

    def lstm_lrp_arras(self, input_data, rel_prev, aggregate=True):
        
        # compute all activations (gates and signals for lstm layer)
        self.get_lstm_states(input_data)

        if len(rel_prev.shape) == 2 and aggregate:
            # print("DO AGGREGATE")
            rel_prev = np.mean(rel_prev, axis=0)
            
        elif len(rel_prev.shape) == 2 and not aggregate:
            # print("DO NOT AGGREGATE")
            rel_prev = rel_prev[-1, :]

        
        # Extract the last cell state
        cT = np.array(self.cell_states)[-1, 0, :] # (timesteps, batch_size, dimensions) -> (dimensions, )

        # Relevance for the final step -> Leila Arras et al. (2019) in the book Explainable AI (Chpt. 11) write:
        # ck is a constant term that is used to redistrubte the relevance across time - 
        # we will take the previous relevance from the layer above and set ck equal to this value  
        ck = rel_prev


        relevance = []
        
        nlower = input_data.shape[2]
        
        # Get the LSTM weights and biases
        W = self.get_weights()[0]  # LSTM weights (D, 4M) - 4 because of 4 gates
        w_i, w_f, w_c, w_o = tf.split(W, num_or_size_splits=4, axis=1) # (D,M), (D,M), (D,M), (D,M)
        
        # Get the LSTM weights and biases
        b = self.get_weights()[2]  # Biases (4M,)
        b_i, b_f, b_c, b_o = tf.split(b, num_or_size_splits=4, axis=0) # (M,), (M,), (M,), (M,)
        
        
        for t in reversed(range(self.timesteps)): 
            zt = np.array(self.cell_input_signal)[t, 0, :] # (timesteps, batch_size, dimensions) -> (dimensions,)
            it = np.array(self.input_gate_activation)[t, 0, :] # (timesteps, batch_size, dimensions) -> (dimensions,)
            
            # compute product between input signal and input gate
            ap = zt * it
            
            
            if(t == self.timesteps - 1):
                # Rp = ak * ck
                Rp = ap * ck 
            else:
                # R_p-T = ( prod_{t=1}^T a_{f-t+1} ) * a_{p-T} * c_k , where c_k = cT
                Rp = ap * np.multiply.reduce(np.array(self.forget_gate_activation)[t:, 0, :]) * ck 
            
            # using linear rule
            relevance.append(lrp_linear(np.array(w_c), np.array(b_c), input_data[0, t, :], zt, Rp, nlower))
        
        return np.array(relevance)


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
            # print("DO AGGREGATE")
            rel_prev = np.mean(rel_prev, axis=0)
            
        elif len(rel_prev.shape) == 2 and not aggregate:
            # print("DO NOT AGGREGATE")
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
            
            # extract the number of nodes in the lower layer - in the case of LSTMS we consider one timestep at a time
            nlower = input_data.shape[2]
            
            # using linear rule
            relevance.append(lrp_linear(np.array(w_c), np.array(b_c), input_data[0, t, :], zt, Rzt, nlower))
        
        
        #print("LRP LSTM - DONE")
        return np.array(relevance)


