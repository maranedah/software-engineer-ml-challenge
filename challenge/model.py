import pandas as pd

from typing import Tuple, Union, List

import numpy as np
import xgboost as xgb
import joblib
from .preprocess import get_period_day, is_high_season, get_min_diff, get_top_10_features
from sklearn.linear_model import LogisticRegression

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
        self._schema = None

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        
        data['period_day'] = data['Fecha-I'].apply(get_period_day)
        data['high_season'] = data['Fecha-I'].apply(is_high_season)
        data['min_diff'] = data.apply(get_min_diff, axis = 1)
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )
        features.to_csv("full_features.csv")
        self._schema = features.columns.tolist()
        features = get_top_10_features(features)
        features.to_csv("features.csv")
        if target_column:
            target = data[target_column].to_frame()
            return features, target
        else:
            return features
        
    

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0/n_y1
        model= xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
        #model = LogisticRegression(class_weight={1: n_y0/len(target), 0: n_y1/len(target)})
        model.fit(features, target)
        self._model = model
        joblib.dump(self, 'model.joblib')

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model:
            y_hat = self._model.predict(features)
        
        return y_hat.tolist()
