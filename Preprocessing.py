import pandas as pd
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def process_data_with_factors(csv_file_path, factor_selection):
    """
    Process factor data and merge it with S&P 500 monthly returns.

    Parameters:
        csv_file_path (str): The file path of the CSV file containing the factor data.
        factor_selection (list): A list of factor names to select from the factor data.

    Returns:
        pd.DataFrame: A DataFrame containing the S&P 500 monthly returns and selected factors.
                      The DataFrame includes columns for 'Date', 'Return', and each factor in the 'factor_selection' list.
                      Additionally, it includes a 'ReturnFactor' column that contains the S&P 500 monthly returns.
                      The DataFrame is indexed by the 'Date' column.
    """
    
    # Read in factor data as pd.dataframe
    factors = pd.read_csv(csv_file_path)

    # Format date column
    factors["date"] = pd.to_datetime(factors["date"])

    # Copy selection of factors
    factors_filtered = factors[["date"] + factor_selection].copy()

    # Remove NA rows
    factors_filtered.dropna(inplace=True)

    # Key the factors on the calendar month so the merge below aligns rows by
    # date rather than by (fragile) positional row order.
    factors_filtered["month"] = factors_filtered["date"].dt.to_period("M")

    # Get date range of available data
    date_range = [factors_filtered["date"].min(), factors_filtered["date"].max()]

    # Download stock data for the SP500 etf SPX
    print("Download data from yahoo finance ...")
    # auto_adjust=False keeps the 'Adj Close' column that current yfinance
    # versions otherwise drop.
    sp500 = yf.download('^SPX', start=date_range[0], end=date_range[1], auto_adjust=False)
    print("Download: DONE ✔️")
    # Resample to monthly frequency ('ME' = month-end) and calculate returns
    sp500 = sp500["Adj Close"].resample('ME').last().pct_change().reset_index()

    # Rename Adj. Close to 'Return' and derive the same monthly key
    sp500.rename(columns={"Adj Close": "Return", "Date": "date"}, inplace=True)
    sp500["month"] = sp500["date"].dt.to_period("M")

    # Merge returns and factors on the monthly key so they stay date-aligned
    sp500 = pd.merge(sp500, factors_filtered.drop(columns="date"),
                     on="month", how="left")

    # Index by date and drop the helper key
    sp500 = sp500.set_index("date").drop(columns="month")

    # Add returns as a feature
    sp500["ReturnFactor"] = sp500["Return"]

    return sp500


def prepare_data_for_training(data, small_window_size, split_ratio=0.8):
    """
    Prepare data for training a machine learning model.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data for training.
        small_window_size (int): The window size for creating batches of data.
        split_ratio (float, optional): The ratio to split the data into training and testing sets.
                                      Default is 0.8 (80% for training, 20% for testing).

    Returns:
        tuple: A tuple containing four numpy arrays (X_train, y_train, X_test, y_test).
               X_train: Input sequences for training.
               y_train: Output labels (response) for training.
               X_test: Input sequences for testing.
               y_test: Output labels (response) for testing.
    """

    # Drop rows with NaN values introduced by shifting
    data.dropna(inplace=True)

    # Create an instance of StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to the data and transform the data
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Define X, y
    X = data.drop(columns="Return").values
    y = data["Return"].values

    # Define the window size for one batch of data
    window_size = small_window_size

    # Create input sequences and response
    sequences = []
    response = []

    for i in range(window_size, len(X)):

        # Take features at (i-win : i)
        sequences.append(X[i-window_size:i])
        # Take response at (i)
        response.append(y[i])

    # Convert the lists to numpy arrays
    sequences = np.array(sequences)
    labels = np.array(response)

    # Split the data into training and testing sets based on the split_ratio
    split_index = int(split_ratio * len(sequences))
    X_train = sequences[:split_index]
    y_train = labels[:split_index]
    X_test = sequences[split_index:]
    y_test = labels[split_index:]

    return X_train, y_train, X_test, y_test
