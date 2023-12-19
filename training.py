import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import os
import mediapipe as mp
import seaborn as sns
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
signs = np.array(['CamOn', 'Cha', 'Chao', 'Me', 'Ong', 'TamBiet'])

number_videos = 100
number_frames = 40
epoch = 400
ti=2 #times
#load data
sequences = np.load("E:/LSTMSignLanguageDetection/data_feature/all_feature.npy")
labels = np.load("E:/LSTMSignLanguageDetection/data_feature/all_label.npy")

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(256, return_sequences=True,input_shape=(40, 134)))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(16))
model.add(Dense(signs.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(f'E:/LSTMSignLanguageDetection/checkpoint/weights.{epoch:02d}_{ti}.hdf5', monitor='val_accuracy', verbose=1, mode = 'max', save_best_only=True, save_weights_only=False, period=1)
history = model.fit(X_train, y_train, epochs=epoch, callbacks=[checkpoint],validation_data=(X_test,y_test))
model.summary()

# plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(f'E:/LSTMSignLanguageDetection/result/model_accuracy_e{epoch}_{ti}.png')
plt.clf()

# plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(f'E:/LSTMSignLanguageDetection/result/lstm_loss_e{epoch}_{ti}.png')

##### Confusion matrix #####
yhat = model.predict(X_test)
yhat = np.argmax(yhat, axis=1).tolist()
ytrue = np.argmax(y_test, axis=1).tolist()
cfm = confusion_matrix(ytrue, yhat)

class_names = ['CamOn', 'Cha', 'Chao', 'Me', 'Ong', 'TamBiet']
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cfm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

acc_score=accuracy_score(ytrue, yhat)
f1_score = f1_score(ytrue, yhat, average='weighted')
recall = recall_score(ytrue, yhat, average='weighted')
precision = recall_score(ytrue, yhat, average='weighted')
print("Evaluate Model:")
print ("-----------------------------------------------------------")
print("  Accuracy:", acc_score)
print("  F1 Score:", f1_score)
print("  Recall:", recall)
print("  Precision:", precision,'\n')
print (checkpoint)
plt.savefig(f'E:/LSTMSignLanguageDetection/result/cfm_e{epoch}_{ti}.png')
model.save(f'E:/LSTMSignLanguageDetection/model/lstm_model_e{epoch}_{ti}.h5')