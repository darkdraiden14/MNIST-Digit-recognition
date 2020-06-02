# Step 1 : Import all the required libraries and Load the dataset
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#dataset
from keras.datasets import mnist

#load the dataset and split it into test and train sets

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#Checkpoint 1
print(X_train.shape, Y_train.shape)

# Step 2 :Preprocessing of data

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
#will store original input shape
input_shape = (28,28,1)

#lets print Y
#print(Y_train)

# Apply One hot encoding on Y_train and Y_test
# for converting class vectors to binary class metrices
Y_train = keras.utils.to_categorical(Y_train, num_classes = 10)
Y_test = keras.utils.to_categorical(Y_test, num_classes = 10)

#convert int into float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /=255
X_test /=255

#Checkpoint 2
print(X_train.shape)

# Step 3: Create the model

num_classes =10

model = Sequential()

#input Layer
model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu', input_shape = input_shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) # inorder to reduce the over fitting while training
model.add(Flatten())

#Hidden Layer
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

#output Layer
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss = keras.losses.categorical_crossentropy, 
optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

# Step 4: Training the model

digitModel = model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=1,
validation_data=(X_test,Y_test))

#Checkpoint 3
print("-----TRAINING SUCCESSFULL-----")

#Now Save the model
model.save('mnist_digit.h5')
print("MODEL SAVED")

# Step 5: Evaluation

accuracy_score = model.evaluate(X_test, Y_test, verbose=0)
print("Test Lose : ", accuracy_score[0])
print("Test Accuracy : ", accuracy_score[1])


# Model Creation is done