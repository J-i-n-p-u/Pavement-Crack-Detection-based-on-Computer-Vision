import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns
import os
import glob
import pandas as pd

class CNNDetector:
    def __init__(self, checkpoint_file, input_shape=(64,64) ):        
        self.input_shape = input_shape
        self.model = load_model(checkpoint_file)        
        
    def predict_image_file(self, filename):        
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(int(self.input_shape[0]),int(self.input_shape[1]))) 
        img_converted = img.reshape(1,self.input_shape[0],self.input_shape[1],3)
        return self.model.predict(img_converted)
    
def pick_random_files(n_files, path, class_names):
    image_files = []
    for name in class_names:
        tmp=os.listdir(path+name)
        for i in range(int(n_files/len(class_names))):
            image_files.append(path+name+"/"+random.choice(tmp))
    random.shuffle(image_files)
    return image_files

def eval_res_helper(CHECKPOINT_FILE, Input_shape, path, image_files, ground_truth):
    
    INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT = Input_shape
    cnn = CNNDetector(CHECKPOINT_FILE, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
    y_true = []
    result = []
    for j, img_filename in enumerate(image_files):
        if j%100 == 0:
            print(j)
        pred = cnn.predict_image_file(path + img_filename)
        result.append(pred)
        y_true.append(ground_truth)
    result = np.array(result)
    y_pred = []
    for i in range(result.shape[0]):
        if result[i][0][0]> 0.5:
            y_pred.append(0)
        else:
            y_pred.append(1)
    return result, y_true, y_pred

def plot_cm(y_test, y_pred, name):
    axis_labels = ["No Cracks","Cracks"]
    cf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (6,5))
    sns.heatmap(cf_matrix/np.sum(cf_matrix, axis = 1, keepdims=True), annot=True, 
                fmt='.2%', cmap='Blues', xticklabels=axis_labels, yticklabels=axis_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title('Confusion Matrix', fontsize=20)
    plt.savefig(name)
    test_score = f1_score(y_test, y_pred, average = 'weighted')
    print('Test Score:', test_score)
    return test_score

def eval_res(CHECKPOINT_FILE, Input_shape, eval_num, test_p_path, test_n_path, random_seed):
    
    test_img_files = []
    
    image_files0 = os.listdir(test_n_path)
    random.seed(random_seed)
    random.shuffle(image_files0)
    image_files0 = image_files0[:eval_num[0]]
    
    [test_img_files.append(test_n_path+name) for name in image_files0]
    result0, y_true0, y_pred0 = eval_res_helper(CHECKPOINT_FILE, Input_shape, test_n_path, image_files0, 0)

    image_files1 = os.listdir(test_p_path)
    random.seed(random_seed)
    random.shuffle(image_files1)
    image_files1 = image_files1[:eval_num[1]]
    [test_img_files.append(test_p_path+name) for name in image_files1]
    result1, y_true1, y_pred1 = eval_res_helper(CHECKPOINT_FILE, Input_shape, test_p_path, image_files1, 1)

    result = np.concatenate([ele.reshape(-1, 2) for ele in [result0, result1]])
    
    data = {'Fig_name': image_files0+image_files1,
            'Fig_path': test_img_files,
            'True Label': y_true0+y_true1,
            'Predicted Label': y_pred0+y_pred1,
            'Non_Cracks Confidence': result[:,0],
            'Cracks Confidence': result[:,1]}
    
    return pd.DataFrame.from_dict(data)

if __name__ == "__main__":
    
    # CHECKPOINT_FILE = r'update-model-checkpoints/v1/simplenet_cracks8020_weights.16-0.24.hdf5'
    # CHECKPOINT_FILE = r'update-model-checkpoints/simplenet_cracks8020_weights.12-0.43.hdf5'
    # CHECKPOINT_FILE = r'update-model-checkpoints/simplenet_cracks8020_weights.70-0.20.hdf5'
    # CHECKPOINT_FILE = r'febrero-cpu-friendly_weights.27-0.01.hdf5'
    # CHECKPOINT_FILE = r'update-model-checkpoints/simplenet_cracks8020_weights.14-0.25.hdf5'
    # CHECKPOINT_FILE = r'update-model-checkpoints/simplenet_cracks8020_weights.19-0.24.hdf5'
    # test_p_path = r'wsp_cracks_splitted8020/test_set/Positive/'
    # test_n_path = r'wsp_cracks_splitted8020/test_set/Negative/'
    
    test_p_path = r'cracks_splitted8020/test_set/Positive/'
    # list_of_images_test_p = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(image_directory_test_p))]
    
    test_n_path = r'cracks_splitted8020/test_set/Negative/'
    # list_of_images_test_n = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(image_directory_test_n))]
    random_seed = 1
    # Input_shape = (64, 64)
    list_of_files = glob.glob('model-checkpoints/*.hdf5') 
    CHECKPOINT_FILE = max(list_of_files, key=os.path.getctime) # last checkpoint
    
    
    Input_shape = (64, 64)
    eval_num = [15, 15]
    # eval_num = [4346, 522]
    df= eval_res(CHECKPOINT_FILE, Input_shape, eval_num, test_p_path, test_n_path, random_seed)
    
    # (y_true, y_pred) = eval_res(i, test_p_path, test_n_path)
    score = plot_cm(df['True Label'], df['Predicted Label'], '_wsp_cm.jpg')
