from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0
x_test = x_test.reshape((x_test.shape[0], -1)) / 255.0

x_val = x_train[:1000]
x_train = x_train[1000:]

y_val = y_train[:1000]
y_train = y_train[1000:]

# Define the RandomForestClassifier with custom hyperparameters
clf = RandomForestClassifier(n_estimators=10, random_state=42)
# clf = RandomForestClassifier(
#     n_estimators=200,  # Number of trees in the forest
#     max_depth=20,       # Maximum depth of each tree
#     min_samples_split=2,  # Minimum number of samples required to split an internal node
#     min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
#     random_state=42
# )

# Train the random forest classifier
clf.fit(x_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(x_train)
accuracy = accuracy_score(x_train, y_pred)
print(f"Validation Accuracy: {accuracy}")

# Test on the testing set
y_test_pred = clf.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy}")
