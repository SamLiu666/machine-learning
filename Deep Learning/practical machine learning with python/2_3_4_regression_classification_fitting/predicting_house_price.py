from keras.datasets import boston_housing
import numpy as np


(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

# prepare the data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# building network
from keras import models, layers


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=['mae'])
    return model


# k-fold validation
def k_fold_validation(k, num_epochs,batch_size):
    # k = 4
    num_val_samples = len(train_data)//k
    # num_epochs = 100
    all_scores = []

    for i in range(k):
        print('processing fold #', i)
        val_data = train_data[i*num_val_samples: (i+1)*num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i+1)*num_val_samples]

        partial_train_data = np.concatenate(
            [train_data[:i*num_val_samples],
             train_data[(i+1)*num_val_samples:]],
            axis=0
        )
        partial_train_targets  = np.concatenate(
            [train_targets[:i*num_val_samples],
             train_targets[(i+1)*num_val_samples:]],
            axis=0
        )

        model = build_model()
        # trian
        # model.fit(partial_train_data, partial_train_targets,
        #           epochs=num_epochs, batch_size=64, verbose=1)
        # # evaluate with val_data
        # val_mse, val_mae = model.evaluate(val_data,val_targets, verbose=1)
        # all_scores.append(val_mae)

        history = model.fit(partial_train_data, partial_train_targets,
                            validation_data=(val_data, val_targets),
                            epochs=num_epochs, batch_size=batch_size, verbose=1)
        mae_history = history.history['val_mae']
        all_scores.append(mae_history)

    return all_scores


from keras import backend as K

# Some memory clean-up
K.clear_session()
num_epochs=500
all_scores = k_fold_validation(k=4, num_epochs=num_epochs, batch_size=64)
average_mae_history  = [np.mean([x[i] for x in all_scores]) for i in range(num_epochs)]
# print("平均得分：", average_mae_history )


# 可视化 mae
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# 平滑处理
def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      # 指数加权平均数
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# Get a fresh, compiled model.
# model = build_model()
# # Train it on the entirety of the data.
# model.fit(train_data, train_targets,
#           epochs=80, batch_size=16, verbose=0)
# test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)