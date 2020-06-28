from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16
from keras import backend as K
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'

def predict(img_path, output_path=None):
    # use VGG16
    K.clear_session()
    model = VGG16(weights='imagenet')

    # `img` is a PIL image of size 224x224, reshape
    img = image.load_img(img_path, target_size=(224, 224))

    # `x` is a float32 Numpy array of shape (224, 224, 3)
    x = image.img_to_array(img)

    # reshape "batch" of size (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)

    # Finally we preprocess the batch (this does channel-wise color normalization)
    x = preprocess_input(x)

    preds = model.predict(x)
    ans = decode_predictions(preds, top=3)[0]
    ans0, ans1 = ans[0][1:], ans[1][1:]
    name, propba = ans0[0], ans0[1]
    if propba >=0.5:
        print('The picture is {} with {:.3} probalility:'.format(name, propba))
    else:
        print(ans)

    # This is the "african elephant" entry in the prediction vector
    african_elephant_output = model.output[:, 386]

    # The is the output feature map of the `block5_conv3` layer,
    # the last convolutional layer in VGG16
    last_conv_layer = model.get_layer('block5_conv3')

    # This is the gradient of the "african elephant" class with regard to
    # the output feature map of `block5_conv3`
    grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

    # This is a vector of shape (512,), where each entry
    # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`,
    # given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the elephant class
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.title(name)
    plt.show()

    if output_path:
        # We use cv2 to load the original image
        img = cv2.imread(img_path)

        # We resize the heatmap to have the same size as the original image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # We convert the heatmap to RGB
        heatmap = np.uint8(255 * heatmap)

        # We apply the heatmap to the original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 0.4 here is a heatmap intensity factor
        superimposed_img = heatmap * 0.4 + img

        # Save the image to disk
        obj_path = output_path + name + ".jpg"
        cv2.imwrite(obj_path, superimposed_img)


if __name__ == '__main__':
    # img_path = r"D:\machine learning\Deep Learning\DL with python\5_CNN_picture\creative_commons_elephant.jpg"
    # output = r'D:\machine learning\Deep Learning\DL with python\5_CNN_picture\vilualization\elephant_cam.jpg'
    dir_path = r"D:\machine learning\0614\coursera_dl_class\DL\1 Neural Networks and Deep Learning\week 4\datasets\\"
    out_dir = r"D:\machine learning\Deep Learning\DL with python\5_CNN_picture\vilualization\\"
    print("\n")
    for i in range(1,5):
        img_path = dir_path + str(i) + ".jpg"
        print(img_path, "prediction: ")
        # out_path = out_dir + str(i) + ".jpg"
        predict(img_path, output_path=None)
