from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Reshaping the data
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
x_test = x_test.reshape((x_test.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3]))

# Normalizing pixel values
x_train /= 255
y_train /= 255

# Splitting first 1000 samples to validation and rest into training
x_test = x_train[:1000]
x_train = x_train[1000:]

y_test = y_train[:1000]
y_train = y_train[1000:]

# Define the RandomForestClassifier with custom hyperparameters
clf = RandomForestClassifier(n_estimators=10, random_state=42)

# Train the random forest classifier
clf.fit(x_train, y_train)

# Evaluate the classifier on the validation set
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)

accuracy = accuracy_score(y_train, y_train_pred)
print(f"Validation Accuracy: {accuracy}")

# Test on the testing set
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy}")
