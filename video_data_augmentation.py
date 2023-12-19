import cv2
import os
import numpy as np


def add_gaussian_noise(frame, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, frame.shape)
    noisy = np.clip(frame + gauss, 0, 255)
    return noisy.astype(np.uint8)

# I need more data at 3 label 'CamOn', 'Cha', 'Chao'
node_dir = ['CamOn', 'Cha', 'Chao']
number_need = [23, 20, 26] #number of video I need to have 100 video each label
for i_folder in range(len(node_dir)):
    path = "E:/Q/tailieu/nam3/ML/VSLDataset/" + node_dir[i_folder]
    obj_list = os.listdir(path)
    for i_file in range (number_need[i_folder]):
        video_path = path + "/" + obj_list[i_file]
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(3))
        height = int(cap.get(4))
        size = (width, height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        file_name = path + "/" + obj_list[i_file].rstrip('.avi') + "_noise" + '.avi' # I also rename it after augmentation
        result_video = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, size, True)
        print(file_name)
        while(cap.isOpened()):
            ret, frame= cap.read()
            if ret == False:
                break
            ##Apply Gaussian noise
            noisy_frame = add_gaussian_noise(frame = frame)
            result_video.write(noisy_frame)

        cap.release()
        result_video.release()
cv2.destroyAllWindows()