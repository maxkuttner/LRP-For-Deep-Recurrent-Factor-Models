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


# Basic Overview
Welcome to the **Deep Recurrent Factor Model** Repository. This repository introduces a fresh approach to deep feed-forward LSTM networks, featuring a new layer class that enables easy layerwise relevance propagation. 

By building upon the familiar `keras.layers` module, this repository allows you to create deep LSTM networks and fascilitate LRP.
The key highlight is the `CustomModel` class, which takes care of the complex task of backpropagating relevance through any variation of custom `Input`,`LSTM`, 
`Dense` or `Dropout` layers, which are built using the `keras` functional API. 

This means you can now design deep feedforward LSTM models and extract feature relevance. Explore this repository to harness enhanced interpretability and customization in your LSTM network designs to produce better and more interpretable return predictions.


# Getting Started

In order to get started clone the GitHub repository to your local machines:
```bash
git clone https://github.com/ACM40960/project-mkaywins.git
```

## Intalling Dependencies
- Make sure to have python 3.11+ installed - if not download the latest version of Python 3 [[here]](https://www.python.org/downloads/).

- Install all necessary dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Build a model

If you want to build your own deep LSTM model, then you need to 
use the [Functional API by Keras](https://keras.io/guides/functional_api/). This is shown in the following example:

```python
# Build an example model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_shape = (5, 16)

#1) Input layer
input_layer = Input(shape=input_shape, name="Input")

#2) LSTM layer
lstm_output2, hidden_state2, cell_state2 = CustomLSTM(units=8, return_sequences=False,
                                                      return_state=True,
                                                      kernel_initializer='glorot_uniform',
                                                      kernel_regularizer=L2(0.01),
                                                      name="CustomLSTM_2")(input_layer)

#3) Dense layer
output_layer = Dense(1, activation='linear',
 kernel_initializer='glorot_uniform', kernel_regularizer=L2(0.01),
  name="Dense_2_Final")(lstm_output2)

# Create the custom model
model = CustomModel(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Print the model summary
model.summary()

```

# Experimentation 

- [Replication of Experiments in <i>'Deep Factor Models'</i>]() 
- You can find the literature review for this project [[here]](./static/LiteratureReview.pdf).

# Data

- We gatherd the factor data from the openly available factor data set provided by [Andrew Y. Chen and Tom Zimmermann](https://www.openassetpricing.com/data/)

-  You can find a description of the factor data [[here]](https://docs.google.com/spreadsheets/d/1WLiuWh4Uq_0wK230yXpczsb_PON0z91e_TAcUtb0rkU/edit?pli=1#gid=312865186)

- How we try to map features from the [Open Asset Pricing Data Set](https://www.openassetpricing.com/data/) to factors used in the [paper on deep factor models]((https://arxiv.org/pdf/1901.11493.pdf)) is described [[here]](./static/Data/FactorDescription.md).

# References

- The relvance propagation algorithm used is described in <a href=https://proceedings.neurips.cc/paper_files/paper/2019/file/16105fb9cc614fc29e1bda00dab60d41-Paper.pdf> Arjona-Medina, J. A., Gillhofer, M., Widrich, M., Unterthiner, T., Brandstetter, J., & Hochreiter, S. (2019). Rudder: Return decomposition for delayed rewards. Advances in Neural Information Processing Systems, 32.</a>. 

- We utilize the linear relevance propagation function by [Leila Arras](https://github.com/ArrasL/LRP_for_LSTM) next to our own methods to conduct LRP for LSTM layers.



