import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping

K.clear_session()


def plot_25_images(train_images,train_labels, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        plt.colorbar()
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

##################################################################################
# Import the Fashion MNIST dataset
print(tf.__version__)
mnist_data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()
print(train_images.shape, "\n", test_images.shape)
# explore the data
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#plot_25_images(train_images,train_labels, class_names)

##################################################################################
print("##################################################################################")
print("trian model")
# process the data
train_images = train_images/255
n=50000
x_valid = train_images[n:]
y_valid = train_labels[n:]
train_images = train_images[:n]
train_labels = train_labels[:n]
test_images = test_images/255
print(train_images.shape, "\n", x_valid.shape, "\n",test_images.shape)
#plot_25_images(train_images,train_labels, class_names)

# build model
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28,28)))  # cannot visualize ?
model.add(layers.Dense(24, activation="relu"))
model.add(layers.Dense(12, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
model.summary()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


early_check = EarlyStopping(patience=7, monitor="val_loss", mode="min")
callbacks = [early_check]
model.fit(train_images, train_labels, validation_split=0.15,
          shuffle=True, epochs=128, batch_size=512, callbacks=callbacks,verbose=1)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
model.save("image_classify_modify.h5")
print("test_loss, test_acc: " ,test_loss, test_acc )
#ann_viz(model, view=True, filename="model.gv")
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

def visualization_model():
    from ann_visualizer.visualize import ann_viz
    from keras.models import Sequential
    from keras.layers import Dense

    example_model = Sequential()
    example_model.add(Dense(12, input_dim=8, activation='relu'))
    example_model.add(Dense(8, activation='relu'))
    example_model.add(Dense(1, activation='sigmoid'))

    ann_viz(example_model, view=True, filename="network.gv")

# visualization_model()
##################################################################################
# make prediction
print("##################################################################################")
print("make prediction")
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0], np.argmax(predictions[0]), test_labels[0], sep="\n")

# Graph this to look at the full set of 10 class predictions
print("##################################################################################")
print("Graph this to look at the full set of 10 class predictions")
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    # x label show the number predicted wrong
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])

    thisplot = plt.bar(range(10), predictions_array)
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


print("##################################################################################")
print("Use the model")
img = test_images[1]
print(img.shape)
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)
predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
np.argmax(predictions_single[0])