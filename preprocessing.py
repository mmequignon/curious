#!/home/pytorch/pytorch/sandbox/bin/python3

import numpy
from sklearn import preprocessing


input_data = numpy.array([
    [5.1, -2.9, 3.3],
    [-1.2, 7.8, -6.1],
    [3.9, 0.4, 2.1],
    [7.3, -9.9, -4.5],
])
print("Used data :")
print(input_data)

# BINARIZATION : Determine a number, each higher value is 1, else are 0
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("binarization")
print(data_binarized)

# MEAN : Remove the mean in order to center values around 0
print("\nRemoving of the mean")
print("BEFORE :")
print(input_data.mean(axis=0))
print(input_data.std(axis=0))
data_scaled = preprocessing.scale(input_data)
print("AFTER :")
print(data_scaled.mean(axis=0))
print(data_scaled.std(axis=0))

# SCALING : highest value is 1 and other values are relative to it
print("\nScaling")
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print(data_scaled_minmax)

# NORMALIZATION
print("\nNormalization")
print("L1 :")
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
print("L2 : ")
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print(data_normalized_l1)
print(data_normalized_l2)
