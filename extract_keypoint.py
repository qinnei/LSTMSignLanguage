import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


#25 landmark from 0(nose) to 24(righ_hip) in mediapipe pose
#21 landmark from 0(wrist) to 20(pinky_tip) in left_hand
#21 landmark from 0(wrist) to 20(pinky_tip) in right_hand

def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for i, res in enumerate(results.pose_landmarks.landmark) if i <= 24]).flatten() if results.pose_landmarks else np.zeros(25*2)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*2)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*2)
    return np.concatenate([pose, lh, rh])

# 25*2 + 21*2 + 21*2 =  134 key point

# Path for exported data, numpy arrays
DATA_PATH = "E:/LSTMSignLanguageDetection/data_feature"

# label
signs = np.array(['CamOn', 'Cha', 'Chao', 'Me', 'Ong', 'TamBiet'])

# number of videos/ label
number_videos = 100

# each video has 10 * 4 =40 frames total (fps video=10, duration each video= 4 second)
number_frames = 40

### This code make folder to save feature
# for sign in signs:
#     for sequence in range(no_sequences):
#         try:
#             os.makedirs(os.path.join(DATA_PATH, sign, str(sequence)))
#         except:
#             pass

for i_file in range(len(signs)):
    path = "E:/Q/tailieu/nam3/ML/VSLDataset/" + signs[i_file]
    obj_list = os.listdir(path)
    label = i_file
    for j in range(len(obj_list)):
        video_path = path + "/" + obj_list[j]
        cap = cv2.VideoCapture(video_path)
        with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
            frame_num = 0
            while (cap.isOpened() and frame_num<number_frames):
                ret, frame = cap.read()
                if ret == False:
                    break
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, signs[i_file], str(j), str(frame_num))
                np.save(npy_path, keypoints)
                frame_num += 1
        cap.release()
cv2.destroyAllWindows()

label_map = {label:num for num, label in enumerate(signs)}
print(label_map)
sequences, labels = [], []
for sign in signs:
    for sequence in range(number_videos):
        window = []
        for frame_num in range(number_frames):
            res = np.load(os.path.join(DATA_PATH, sign, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[sign])
npy_path1 = os.path.join(DATA_PATH, "all_feature")
np.save(npy_path1, sequences)

npy_path2 = os.path.join(DATA_PATH, "all_label")
np.save(npy_path2, labels)