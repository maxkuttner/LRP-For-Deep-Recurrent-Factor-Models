import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.callbacks import Callback, EarlyStopping
from keras.regularizers import L2
from CustomLayers import CustomLSTM
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


# LRP
from LRPMethods import lrp_linear
print("LSTM Load Dependencies: DONE ✔️\n\n")

# Define your custom callback to capture activations
class ActivationLogger(Callback):
    def __init__(self):
        super().__init__()
        self.activations = {}

    def capture_activations(self, model, input_data):
        for i, layer in enumerate(model.layers):
            layer_output = layer.output
            activation_model = Model(inputs=model.input, outputs=layer_output)
            activations = activation_model.predict(input_data, verbose=False)
            self.activations[i] = self.add_activations(activations, layer)
    
    def add_activations(self, activations, layer):
        if isinstance(layer, CustomLSTM):
            
            output, hidden_state, cell_state = activations
            
            return {"output": output,
                     "hidden_state": hidden_state,
                     "cell_state": cell_state}
        else:
            return {"output": activations}


# Override the layers property to include the intermediate outputs
class CustomModel(tf.keras.Model):
    def __init__(self, inputs, outputs):
        
        # constructor of tf.keras.Model class
        super().__init__(inputs, outputs)
    
    @property
    def layers(self):
        # exclude Dropout from the list of layers - makes it easier to do LRP
        return [layer for layer in super().layers if not isinstance(layer, Dropout)]

    def backpropagate_relevance(self, input_data, aggregate, type="arras"):
        
        # log activations of network
        activation_logger = ActivationLogger()
        activation_logger.capture_activations(self, input_data)

        # initialise relevance score Rj with the activation of the final output
        Rj = activation_logger.activations[len(self.layers) - 1]["output"][0]
        
        # iterate over the layers of the network in reversed order
        for i in reversed(range(len(self.layers))):

            # get current layer
            current_layer = self.layers[i]

            # skip drouput layer
            if isinstance(current_layer, Dropout):
                continue
            
            # handle: Dense layer
            if isinstance(current_layer, Dense):
                
                # get the number of nodes in the lower layer as this number is used in the lrp rule
                nlower = activation_logger.activations[i - 1]["output"].shape[1]

                # get weights and biases of current layer
                w = current_layer.get_weights()[0]
                b = current_layer.get_weights()[1]

                # check if last layer
                if i == len(self.layers) - 1:
                    # if last layer, then use activation/Rj as nodes for the higher layer
                    zj = Rj
                else:
                    # else use the output of the current layer as nodes for the higher layer
                    zj = activation_logger.activations[i]["output"][0, :]

                # get activation from lower layer
                zi = activation_logger.activations[i - 1]["output"][0, :]

                # user linear lrp rule
                Rj = lrp_linear(w, b, zi, zj, Rj, nlower)

            # hanlde: Custom LSTM layer
            elif isinstance(current_layer, CustomLSTM):
                
                if i == 0:
                    # choose type of LRP rule
                    if type == "arras":
                        Rj = current_layer.lstm_lrp_arras(input_data, Rj, aggregate)
                    else:
                        Rj = current_layer.lstm_lrp_rudder(input_data, Rj, aggregate)
                else:
                    
                    # get input from end of previous layer
                    input_tmp = activation_logger.activations[i - 1]["output"]
                    
                    # choose type of LRP rule
                    if type == "arras":
                        Rj = current_layer.lstm_lrp_arras(input_tmp, Rj, aggregate)
                    else:
                        Rj = current_layer.lstm_lrp_rudder(input_tmp, Rj, aggregate)

        return Rj


def rolling_fit(input_layer, output_layer, 
                X_train, y_train,
                big_window_size=60,
                validation_size=1/60, 
                do_plot=True, save_plot=True):

    num_windows = len(X_train) - big_window_size 
    validation_set_size = int(num_windows * validation_size)

    predictions = []
    relevance = []

    tmp_model = CustomModel(inputs=input_layer, outputs=output_layer) 
    init_weights = tmp_model.get_weights()
    tmp_model.summary()
    
    del tmp_model
    
    print("\nStarting rolling window fitting...\n")
    for i in tqdm(range(num_windows)):
        start_index = i
        end_index = i + big_window_size

        X_train_window = X_train[start_index:end_index]
        y_train_window = y_train[start_index:end_index]
        y_train_window = y_train_window.reshape(-1, 1)

        X_val = X_train[end_index : end_index + validation_set_size]
        y_val = y_train[end_index : end_index + validation_set_size]
        y_val = y_val.reshape(-1, 1)
        
        # Create an instance of your CustomModel
        model = CustomModel(inputs=input_layer, outputs=output_layer)  # Adjust this with your input and output layers
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Define the EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=0)

        # Train the model on the current rolling window
        model.fit(X_train_window,
                y_train_window,
                validation_data=(X_val, y_val),
                epochs=10,
                batch_size=32,
                verbose=False,
                callbacks=[early_stopping])

        X_next = X_train[end_index:end_index + 1]  # Get the last observation in the validation set

        next_pred = model.predict(X_next, verbose=False)
        predictions.append(next_pred)

        rel = model.backpropagate_relevance(X_next, False)
        relevance.append(rel)
        
        model.set_weights(init_weights) # reinitialise weights

    predictions = np.array(predictions)[:, 0, 0]

    if do_plot:
        plt.plot(y_train[big_window_size:], label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.xlabel('Timestep')
        plt.ylabel('Response')
        plt.legend()
        if save_plot:
            plot_dir = './static/images/rolling_fit'
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, 'rolling_fit_plot.png')
            plt.savefig(plot_path)
        plt.show()
        plt.show()

    print("Rolling window fitting completed.")
    return predictions, relevance



if __name__ == '__main__':
    #Testing
    
    # ----------------------------- MODEL CONSTRUCTION ----------------------------------
    timesteps = 5
    input_dim = 16
    input_shape = (timesteps, input_dim)

    # Create input layer
    input_layer = Input(shape=input_shape, name="Input")
    # Create a CustomLSTM layer
    lstm_output, hidden_state, cell_state = CustomLSTM(units=32, return_sequences=False,
                                                    return_state=True,
                                                    kernel_regularizer=L2(0.02),
                                                    name="CustomLSTM_1")(input_layer)
    # Apply dropout to LSTM output
    dropout1 = Dropout(0.2)(lstm_output)
    # Create a Dense layer
    dense1 = Dense(16, kernel_regularizer=L2(0.02), name="Dense_1")(dropout1)
    # Apply dropout to dense1 output
    dropout2 = Dropout(0.2)(dense1)
    # Create the final output layer
    output_layer = Dense(1, kernel_regularizer=L2(0.02), name="Dense_2_Final")(dropout2)
    # ----------------------------- END: MODEL CONSTRUCTION ----------------------------------



    # Create an instance of your custom model
    custom_model = CustomModel(inputs=input_layer, outputs=output_layer)

    # Compile the model
    custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # # Generate some example training data (replace this with your actual data)
    import numpy as np
    num_samples = 100
    X_train = np.random.rand(num_samples, timesteps, input_dim)
    y_train = np.random.rand(num_samples, 1)

    # # Train the model
    custom_model.fit(X_train, y_train, epochs=10, batch_size=32)



    # After training, capture activations using the callback
    batch_size = 1  # Number of samples in a batch
    timesteps = 5  # Number of time steps
    input_dim = 16  # Dimensionality of each input
    input_data = np.random.rand(1, timesteps, input_dim) # sample input


    print(custom_model.backpropagate_relevance(input_data, False))



