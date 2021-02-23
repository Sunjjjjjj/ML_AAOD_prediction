import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow import keras
#### 模拟数据生成函数：多个正弦波+随机噪音
def generate_time_series(batch_size, n_steps,seed=10):
    np.random.seed(seed)
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5) # + noise
    return series[..., np.newaxis].astype(np.float32)
#### 生成模拟时间序列以及训练集和测试集
n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, Y_test = series[9000:, :n_steps], series[9000:, -1]

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(X_valid[0, :]) 
ax.scatter(50, Y_valid[0], marker = '+', s = 20, c = 'k')
#%%
####朴素预测方法（naive forecast）作为预测效果比较
Y_pred = X_valid[:, -1]
np.mean(keras.losses.mean_squared_error(Y_valid, Y_pred))
ax.scatter(50, Y_pred[0], marker = '+', s = 20, c = 'r')
#%%
#### linear regression
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50, 1]),
    keras.layers.Dense(1)
])
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(0.01))
model.fit(X_train,Y_train,epochs=20,verbose=0)
model.evaluate(X_valid,Y_valid)
Y_pred = model.predict(X_valid)
ax.scatter(50, Y_pred[0], marker = '+', s = 20, c = 'g')
#%%
#### simple RNN
model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1])
])
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(0.01))
model.fit(X_train,Y_train,epochs=20,verbose=0)
model.evaluate(X_valid,Y_valid)
Y_pred = model.predict(X_valid)
ax.scatter(50, Y_pred[0], marker = '+', s = 20, c = 'y')
#%%
#### deep rnn #1
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1)
])
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(0.01))
model.fit(X_train,Y_train,epochs=20,verbose=0)
model.evaluate(X_valid,Y_valid)
Y_pred = model.predict(X_valid)
ax.scatter(50, Y_pred[0], marker = '+', s = 20, c = 'm')

#%%
#### deep RNN:LSTM
t1 = time.time()
model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20),
    keras.layers.Dense(1)
])
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(0.01))
model.fit(X_train,Y_train,epochs=20,verbose=1)
model.evaluate(X_valid,Y_valid)
Y_pred = model.predict(X_valid)
ax.scatter(50, Y_pred[0], marker = '+', s = 20, c = 'm')

t2 = time.time()
print('Time used: %1.2f s' % (t2 - t1))