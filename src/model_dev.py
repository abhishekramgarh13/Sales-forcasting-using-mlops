import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor

class Model(ABC):
    # abstract class for all models

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train : Training data
            y_train: Training lable
        Returns:
            None
        """
        pass

class RandomForestModel(Model):
    """Random Forest Model"""

    
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train : Training data
            y_train: Training lable
        Returns:
            None
        """
    
        try:
            reg = RandomForestRegressor(n_estimators=90,oob_score=True,random_state=42,**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training the model: {}".format(e))
            raise e 
        