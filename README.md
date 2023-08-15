<p align="center"><img width=% src="./static/logo.png" style="margin-bottom=0px"></p>
<div align="center">

![Static Badge](https://img.shields.io/badge/Python-3.11-green?style=flat-square&logo=python&logoColor=%23fff)
![Static Badge](https://img.shields.io/badge/Jupyter-1.0-green?style=flat-square&logo=jupyter&logoColor=%23fff)
![Static Badge](https://img.shields.io/badge/Tensorflow-2.13.0-orange?style=flat-square&logo=tensorflow&logoColor=%23fff)
[![License](https://img.shields.io/badge/license-MIT-red?style=flat-square)](./License)

</div>

---

> **Deep Recurrent Factor Model** is a term coined by the authors of the paper [*Deep Recurrent Factor Model: Interpretable Non-Linear and Time-Varying
> Multi-Factor Model*](https://arxiv.org/pdf/1901.11493.pdf). The authors challenge the idea of linear factor models to predict stock returns and use Long-Short-Term Memory networks (LSTM) in conjunction with layer-wise-relevance propagation (LRP) to construct a time-varying factor model that outperforms equivalent linear models, whilst providing insights into the relvance of particular factors in the prediction.



<div align=center>
This repository provides <b>classes</b>, <b>functions</b> and <b>notebooks</b>
to test <b>Deep Recurrent Factor Models</b> on the US-Stock market.
</div>

  
  

---

  

# [Contents](#contents)
 - [Basic Overview](#basic-overview)
 - [Getting Started](#getting-started) 
    - [Installing Dependencies](#intalling-dependencies)
    - [Build a model](#build-a-model)
    - [LRP](#lrp)
 - [Example](#example)
 - [Data](#data)
 - [References](#references)
 - [Contact](#contact)


# [Basic Overview](#basic-overview)
Welcome to the **Deep Recurrent Factor Model** Repository. This repository introduces a fresh approach to deep feed-forward LSTM networks, featuring a new layer class that enables easy layerwise relevance propagation. 

By building upon the familiar `keras.layers` module, this repository allows you to create deep LSTM networks and fascilitate LRP.
The key highlight is the `CustomModel` class, which takes care of the complex task of backpropagating relevance through any variation of custom `Input`,`LSTM`, 
`Dense` or `Dropout` layers, which are built using the `keras` functional API. 

This means you can now design deep feedforward LSTM models and extract feature relevance. We explore and replicate the approach suggested in the paper [*Deep Recurrent Factor Model: Interpretable Non-Linear and Time-Varying Multi-Factor Model*](https://arxiv.org/pdf/1901.11493.pdf) to test our implementation on the US stock market.


# [Getting Started](#getting-started)

In order to get started clone the GitHub repository to your local machines:
```bash
git clone https://github.com/ACM40960/project-mkaywins.git
```

## [Intalling Dependencies](#intalling-dependencies)
- Make sure to have python 3.11+ installed - if not download the  [latest version of Python 3](https://www.python.org/downloads/).

- Install all necessary dependencies:

    ```bash
    cd ./project-mkaywins
    pip install -r requirements.txt
    ```

## [Build a model](#build-a-model)

If you want to build your own deep LSTM model, then you need to 
use the [Functional API by Keras](https://keras.io/guides/functional_api/). This is shown in the following example:

```python
# Build an example model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

timesteps = 5
input_dim = 16
input_shape = (timesteps, input_dim)

#1) Create input layer
input_layer = Input(shape=input_shape, name="Input")

#2) Create a CustomLSTM layer
lstm_output, hidden_state, cell_state = CustomLSTM(units=32, return_sequences=False,
                                                return_state=True,
                                                kernel_regularizer=L2(0.02),
                                                name="CustomLSTM_1")(input_layer)
#3) Apply dropout to LSTM output
dropout1 = Dropout(0.2)(lstm_output)

#4) Create a Dense layer
dense1 = Dense(16, kernel_regularizer=L2(0.02), name="Dense_1")(dropout1)

#5) Apply dropout to dense1 output
dropout2 = Dropout(0.2)(dense1)

#6) Create the final output layer
output_layer = Dense(1, kernel_regularizer=L2(0.02), name="Dense_2_Final")(dropout2)

# Create an instance of your custom model
custom_model = CustomModel(inputs=input_layer, outputs=output_layer)

# Compile the model
custom_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# # Generate some example training data (replace this with your actual data)
num_samples = 100
X_train = np.random.rand(num_samples, timesteps, input_dim)
y_train = np.random.rand(num_samples, 1)

# Train the model
custom_model.fit(X_train, y_train, epochs=10, batch_size=32)

```

## [Layerwise Relevance Propagation (LRP)](#lrp)

![](./static/images/readme/linearerem-1.jpg)

![](./static/images/readme/lstmlrp-2.jpg)

After having fit the model either by the customary `model.fit()` method, we can proceed to compute the relevance for each input feature. Note, as the input to the model was of dimensions `(timesteps, input_dim) = (5, 16)`, the relevance will have the same dimensions.



```python
# After training, capture activations using the callback
batch_size = 1  # Number of samples in a batch
timesteps = 5  # Number of time steps
input_dim = 16  # Dimensionality of each input
input_data = np.random.rand(1, timesteps, input_dim) # sample input


print(custom_model.backpropagate_relevance(input_data, False))

#>> [[ 7.09338023e-02  1.13306827e-01  4.60653321e-02 -1.32853039e-04
#>>   -1.35259608e-01  8.42626119e-04  8.51461191e-02  2.14990067e-02
#>>    8.02944509e-02  2.89641364e-02  3.84779541e-02 -2.81626266e-02
#>>    1.97666840e-02  4.25549791e-02  4.48795180e-02 -5.27667826e-03]
#>>  [ 2.54032453e-02  1.86068629e-02  6.91068800e-02 -2.39760123e-03
#>>   -1.79765880e-01  6.25969146e-04  1.74778275e-01  1.06237872e-01
#>>    2.17789945e-01  8.48741972e-02  6.93455854e-02 -1.47983451e-02
#>>    2.01158930e-03  4.01856118e-02  1.46082137e-02 -1.32700473e-02]
#>>  [ 1.03305922e-01  1.57469997e-02  4.28371149e-02 -2.59550590e-03
#>>   -1.04792334e-01 -4.62834848e-04  1.36156610e-01  1.08341661e-01
#>>    6.64839048e-02  2.61730689e-02  1.33216296e-01 -2.02558013e-02
#>>    3.66131254e-04  2.98548716e-02  6.76710617e-02 -2.87605669e-02]
#>>  [ 7.00535927e-02  1.08678642e-01  1.83934577e-02 -3.58809669e-03
#>>   -1.83508667e-01  2.84736138e-02  1.32978661e-01  2.21583850e-02
#>>    8.78448212e-02  1.44823108e-02  5.51481198e-02 -3.78624253e-03
#>>    1.50802589e-03  4.10771157e-04  1.67296142e-02 -2.33838079e-02]
#>>  [ 1.32499796e-01  1.38007175e-01  5.90701240e-02  2.62207972e-03
#>>   -3.57363168e-02  1.00417424e-02  2.96777620e-02  7.84326535e-02
#>>    1.12826649e-01  5.46357997e-02  1.93561124e-01 -1.51615638e-02
#>>    1.48435432e-02  3.67880910e-02  3.77769256e-02 -3.78735269e-02]]
```

To summarise the relevance across the `timesteps`, we take the average for each input factor across time.

```python
relevance_aggregated = np.mean(np.array(relevance), axis=1)

#>>  array([0.02649373, 0.0383339 , 0.03583041, 0.02141208, 0.05075075])
```

# [Example](#example)

- [Replication of Experiments in <i>'Deep Factor Models'</i>](./Notebooks/DeepFactorModels.ipynb)
# [Data](#data)

- We gatherd the factor data from the openly available factor data set provided by [Andrew Y. Chen and Tom Zimmermann](https://www.openassetpricing.com/data/)

-  You can find a description of the factor data [[here]](https://docs.google.com/spreadsheets/d/1WLiuWh4Uq_0wK230yXpczsb_PON0z91e_TAcUtb0rkU/edit?pli=1#gid=312865186)

- How we try to map features from the [Open Asset Pricing Data Set](https://www.openassetpricing.com/data/) to factors used in the [paper on deep factor models]((https://arxiv.org/pdf/1901.11493.pdf)) is described [[here]](./static/Data/FactorDescription.md).

# [References](#references)

- The relvance propagation algorithm used is described in <a href=https://proceedings.neurips.cc/paper_files/paper/2019/file/16105fb9cc614fc29e1bda00dab60d41-Paper.pdf> Arjona-Medina, J. A., Gillhofer, M., Widrich, M., Unterthiner, T., Brandstetter, J., & Hochreiter, S. (2019). Rudder: Return decomposition for delayed rewards. Advances in Neural Information Processing Systems, 32.</a>. 

- We utilize the linear relevance propagation function by [Leila Arras](https://github.com/ArrasL/LRP_for_LSTM) next to our own methods to conduct LRP for LSTM layers.


---

# [Contact](#contact)

- Alissia Hrustsova:  alissia.hrustsova@ucdconnect.ie
- Maximilian Kuttner: maximilian.kuttner@ucdconnect.ie
