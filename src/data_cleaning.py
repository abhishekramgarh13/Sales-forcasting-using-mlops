import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """
    def handle_data(self, data : pd.DataFrame)-> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
            data['month']=data['Date'].dt.month
            data['Quarter'] = data['Date'].dt.quarter
            # Convert the 'Date' column to a datetime object
            data['Date'] = pd.to_datetime(data['Date'])
            # Get ISO calendar week number
            data['Week_Number'] = data['Date'].dt.isocalendar().week
            # Define a function to apply the seasonal logic
            def assign_seasonal_value(month):
                if month == 11:
                    return 1
                elif month == 12:
                    return 2
                else:
                    return 0

            # Apply the function to create the 'Seasonal' column
            data['Seasonal'] = data['month'].apply(assign_seasonal_value)
            data = data.drop(['Date','Fuel_Price' ,'Temperature',], axis=1)
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data : {e}")
            raise e 

class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing the data into train and test
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divied the data into train and test 
        """
        try:
            X=data.drop(["Weekly_Sales"],axis=1)
            y=data['Weekly_Sales']
            X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e


class DataCleaning:
    """
    Class for cleaning data which processes the data and divides it into train and test
    """
    def __init__(self, data:pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """
        try: 
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e 

