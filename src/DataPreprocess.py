# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt


class DataPreprocess(object):
    """docstring for DataProcess"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.dataLoader()

    def dataLoader(self):
        rides = pd.read_csv(self.data_path)
        dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
        for each in dummy_fields:
            dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
            rides = pd.concat([rides, dummies], axis=1)
        fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                          'weekday', 'atemp', 'mnth', 'workingday', 'hr']
        data = rides.drop(fields_to_drop, axis=1)
        quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
        # Store scalings in a dictionary so we can convert back later
        scaled_features = {}
        for each in quant_features:
            mean, std = data[each].mean(), data[each].std()
            scaled_features[each] = [mean, std]
            data.loc[:, each] = (data[each] - mean) / std

        # Save the last 21 days
        test_data = data[-21 * 24:]
        data = data[:-21 * 24]
        # Separate the data into features and targets
        target_fields = ['cnt', 'casual', 'registered']
        features, targets = data.drop(target_fields, axis=1), data[target_fields]
        self.test_features, self.test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
        # Hold out the last 60 days of the remaining data as a validation set
        self.train_features, self.train_targets = features[:-60 * 24], targets[:-60 * 24]
        self.val_features, self.val_targets = features[-60 * 24:], targets[-60 * 24:]


if __name__ == '__main__':
    data_path = '../Bike-Sharing-Dataset/hour.csv'
    dp = DataPreprocess(data_path)
    print dp.train_features
