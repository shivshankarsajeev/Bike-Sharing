import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessin_file_obj_path = os.path.join("artifacts", "preprocessing.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This fuction is responsible for data transformation
        """

        try:
            categorical_cols = [
                "season",
                "yr",
                "mnth",
                "hr",
                "holiday",
                "weekday",
                "workingday",
                "weathersit",
            ]
            nominal_cols = ["holiday", "weathersit", "workingday"]
            ordinal_cols = ["season", "mnth", "weekday"]
            ordinal_categories = [
                ["springer", "summer", "fall", "winter"],
                [
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                ],
                [
                    "Sunday",
                    "Monday",
                    "Tueday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                ],
            ]
            numerical_cols = ["temp", "atemp", "hum", "windspeed"]

            # num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

            nominal_pipeline = Pipeline(
                steps=[("onehot", OneHotEncoder(sparse_output=False, drop="first"))]
            )

            ordinal_pipeline = Pipeline(
                steps=[("ordinal", OrdinalEncoder(categories=ordinal_categories))]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    # ("num", num_pipeline, numerical_cols),
                    ("nominal", nominal_pipeline, nominal_cols),
                    ("ordinal", ordinal_pipeline, ordinal_cols),
                ],
                remainder="passthrough",
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test complete")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "cnt"
            train_test_drop = [
                "Unnamed: 0",
                "instant",
                "dteday",
                "casual",
                "registered",
                "cnt",
            ]

            input_feature_train_df = train_df.drop(columns=train_test_drop, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=train_test_drop, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessin_file_obj_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessin_file_obj_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
