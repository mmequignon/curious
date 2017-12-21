#!/home/pytorch/pytorch/sandbox/bin/python3


from sklearn import preprocessing

input_labels = ["red", "black", "red", "green", "black", "yellow", "white"]

print("Labels :", "\n", input_labels, "\n")

encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

print("Correspondences : ")
for i, item in enumerate(encoder.classes_):
    print(item, '->', i)

test_labels = ["green", "red", "black"]
encoded_values = encoder.transform(test_labels)
print(
    "\n", "labels = ", test_labels,
    "→", "Encoded values = ", list(encoded_values))

test_values = [3, 0, 4, 1]
decoded_labels = encoder.inverse_transform(test_values)
print(
    "\n", "values = ", test_values,
    "→", "Decoded labels = ", list(decoded_labels))
