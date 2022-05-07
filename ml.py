import random

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.externals import joblib

import os

SVM_MODEL_NAME = 'svm_model.m'
MLP_MODEL_NAME = 'mlp_model.m'
def generate_outer(mode, num, length):
    for i in range(num):
        x = random.randint(1,length-1)
        y = random.randint(1,length-1)
        img_t = np.zeros(shape=(length,length), dtype = np.uint8)
        img_t[0:x,0:y] = 255

        img = Image.fromarray(img_t)
        img.save(mode+'_corner/Outer_'+str(i)+'_0.png')

        img_t_rot90 = np.rot90(img_t, 1)
        img_rot90 = Image.fromarray(img_t_rot90)
        img_rot90.save(mode+'_corner/Outer_'+str(i)+'_90.png')

        img_t_rot180 = np.rot90(img_t, 2)
        img_rot180 = Image.fromarray(img_t_rot180)
        img_rot180.save(mode+'_corner/Outer_'+str(i)+'_180.png')

        img_t_rot270 = np.rot90(img_t, 3)
        img_rot270 = Image.fromarray(img_t_rot270)
        img_rot270.save(mode+'_corner/Outer_'+str(i)+'_270.png')

        img_t_up_down = img_t[::-1]
        img_up_down = Image.fromarray(img_t_up_down)
        img_up_down.save(mode+'_corner/Outer_'+str(i)+'_ud.png')

        img_t_left_right = np.rot90(img_t, 2)[::-1]
        img_left_right = Image.fromarray(img_t_left_right)
        img_left_right.save(mode+'_corner/Outer_'+str(i)+'_lf.png')

def generate_inner(mode, num, length):
    for i in range(num):
        x = random.randint(1,length-1)
        y = random.randint(1,length-1)
        img_t = np.ones(shape=(length,length), dtype = np.uint8) * 255
        img_t[0:x,0:y] = 0

        img = Image.fromarray(img_t)
        img.save(mode+'_corner/Inner_'+str(i)+'_0.png')

        img_t_rot90 = np.rot90(img_t, 1)
        img_rot90 = Image.fromarray(img_t_rot90)
        img_rot90.save(mode+'_corner/Inner_'+str(i)+'_90.png')

        img_t_rot180 = np.rot90(img_t, 2)
        img_rot180 = Image.fromarray(img_t_rot180)
        img_rot180.save(mode+'_corner/Inner_'+str(i)+'_180.png')

        img_t_rot270 = np.rot90(img_t, 3)
        img_rot270 = Image.fromarray(img_t_rot270)
        img_rot270.save(mode+'_corner/Inner_'+str(i)+'_270.png')

        img_t_up_down = img_t[::-1]
        img_up_down = Image.fromarray(img_t_up_down)
        img_up_down.save(mode+'_corner/Inner_'+str(i)+'_ud.png')

        img_t_left_right = np.rot90(img_t, 2)[::-1]
        img_left_right = Image.fromarray(img_t_left_right)
        img_left_right.save(mode+'_corner/Inner_'+str(i)+'_lf.png')

def generate_other(mode, num, length):
    for i in range(num):
        x = random.randint(0,length)
        if i%2 == 0:
            x = random.randint(0,1)*255
        img_t = np.ones(shape=(length,length), dtype = np.uint8) * 255
        img_t[0:x,0:length] = 0

        img = Image.fromarray(img_t)
        img.save(mode+'_corner/Other_'+str(i)+'_0.png')

        img_t_rot90 = np.rot90(img_t, 1)
        img_rot90 = Image.fromarray(img_t_rot90)
        img_rot90.save(mode+'_corner/Other_'+str(i)+'_90.png')

        img_t_rot180 = np.rot90(img_t, 2)
        img_rot180 = Image.fromarray(img_t_rot180)
        img_rot180.save(mode+'_corner/Other_'+str(i)+'_180.png')

        img_t_rot270 = np.rot90(img_t, 3)
        img_rot270 = Image.fromarray(img_t_rot270)
        img_rot270.save(mode+'_corner/Other_'+str(i)+'_270.png')

        img_t_up_down = img_t[::-1]
        img_up_down = Image.fromarray(img_t_up_down)
        img_up_down.save(mode+'_corner/Other_'+str(i)+'_ud.png')

        img_t_left_right = np.rot90(img_t, 2)[::-1]
        img_left_right = Image.fromarray(img_t_left_right)
        img_left_right.save(mode+'_corner/Other_'+str(i)+'_lf.png')

def generate_inf():
    img_files=os.listdir('img_layout')
    for file in img_files:
        img = Image.open('img_layout/'+file)
        img_crop = img.crop((-1000,-1000,2000,10000))
        img_crop.save('img_layout/'+os.path.splitext(file)[0]+'_crop'+os.path.splitext(file)[1])
    base_x = 990
    step_x = 290
    base_y = 3640
    step_y = 250
    length = 32
    target_img = Image.open('img_layout/gds2img_inv1_0_crop.png')
    for i in range(3):
        for j in range(3):
            target = target_img.crop((base_x+step_x*i, base_y+step_y*j, base_x+step_x*i+length, base_y+step_y*j+length))
            target.save('inference_corner/target_'+str(i)+str(j)+'.png')

def generate_dataset_corner():
    generate_outer(mode = 'train', num = 400, length = 32)
    generate_inner(mode = 'train', num = 400, length = 32)
    generate_other(mode = 'train', num = 400, length = 32)

    generate_outer(mode = 'test', num = 50, length = 32)
    generate_inner(mode = 'test', num = 50, length = 32)
    generate_other(mode = 'test', num = 50, length = 32)
    generate_inf()

def svm_learn():
    x_train=[]
    y_train=[]
    for type in ['Outer', 'Inner', 'Other']:
        for i in range(400):
            for trans in ['_0', '_90', '_180', '_270', '_lf', '_ud']:
                img = Image.open('train_corner/'+type+'_'+str(i)+trans+'.png')
                x_train.append(np.array(img).flatten())
                y_train.append(type)
    clf = svm.SVC(C=50, decision_function_shape='ovr',gamma='scale')
    clf.fit(x_train, y_train)
    joblib.dump(clf, SVM_MODEL_NAME)
    print("Saved model in "+SVM_MODEL_NAME+".")

    x_test=[]
    y_test=[]
    for type in ['Outer', 'Inner', 'Other']:
        for i in range(50):
            for trans in ['_0', '_90', '_180', '_270', '_lf', '_ud']:
                img = Image.open('test_corner/'+type+'_'+str(i)+trans+'.png')
                x_test.append(np.array(img).flatten())
                y_test.append(type)
    print("train acc:", clf.score(x_train, y_train))
    print("test acc:", clf.score(x_test, y_test))

def mlp_learn():
    x_train=[]
    y_train=[]
    for type in ['Outer', 'Inner', 'Other']:
        for i in range(400):
            for trans in ['_0', '_90', '_180', '_270', '_lf', '_ud']:
                img = Image.open('train_corner/'+type+'_'+str(i)+trans+'.png')
                x_train.append(np.array(img).flatten())
                y_train.append(type)
    x_test=[]
    y_test=[]
    for type in ['Outer', 'Inner', 'Other']:
        for i in range(50):
            for trans in ['_0', '_90', '_180', '_270', '_lf', '_ud']:
                img = Image.open('test_corner/'+type+'_'+str(i)+trans+'.png')
                x_test.append(np.array(img).flatten())
                y_test.append(type)

    mlp_clf_tuned_parameters = {"hidden_layer_sizes": [(100, 50), (128, 64)],
                                    "solver": ['adam'],
                                    "max_iter": [80, 100],
                                    "verbose": [True]
                                    }
    mlp = MLPClassifier()
    estimator = GridSearchCV(mlp, mlp_clf_tuned_parameters, n_jobs=6)
    estimator.fit(x_train, y_train)
    joblib.dump(estimator, MLP_MODEL_NAME)
    print("Saved model in "+MLP_MODEL_NAME+".")    
    print(estimator.get_params().keys())
    print(estimator.best_params_)
    # print((clf.predict(x_test)))
    print("train acc:", estimator.score(x_train, y_train))
    print("test acc:", estimator.score(x_test, y_test))

def svr_learn():
    X_train = np.random.randint(200,400,size=(100,1),dtype=np.int)
    print(X_train)
    y_train = X_train.ravel()
    svr = svm.SVR(gamma='scale', C=1e3, kernel='linear')
    # svr = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr.fit(X_train, y_train)

    X_test = np.random.randint(200,400,size=(100,1),dtype=np.int)
    y_test = X_test.ravel()
    print("train acc:", svr.score(X_train, y_train))
    print("test acc:", svr.score(X_test, y_test))

def generate_dataset_seg(mode = 'train', num = 1000):
    for n in range(num):
        edge_len = np.random.randint(1,200) * 40
        print(edge_len)
        gap1 = np.random.randint(1,32,dtype=np.int)
        corner_1 = np.zeros(shape = (gap1,32,3), dtype=np.uint8)
        edge = np.zeros(shape = (edge_len,32,3), dtype=np.uint8)
        gap2 = np.random.randint(1,32,dtype=np.int)
        corner_2 = np.zeros(shape = (gap2,32,3), dtype=np.uint8)

        corner_1[:] = 0 if np.random.randint(0,2) == 0 else 255
        side = 0 if np.random.randint(0,2) == 0 else 1
        border = np.random.randint(1,32,dtype=np.int)
        if side == 0:
            edge[:, 0:border] = 255
        else:
            edge[:, border:32] = 255
        corner_2[:] = 0 if np.random.randint(0,2) == 0 else 255
        if edge_len < 100:
            seg_points = np.array([0, edge_len-1])  # actually no seg
            # print(seg_points)
        elif 100 <= edge_len < 500:
            seg_points = np.array([0, edge_len//2-1, edge_len-1])  # half half rule
            # print(seg_points)
        elif 500 < edge_len:
            in_end = 100
            out_end = 150
            eslen = 200
            seg_points_list = [0, in_end-1, edge_len-out_end-1, edge_len-1]
            for i in range((edge_len-in_end-out_end)//eslen-1):
                separation_len = in_end + (i+1) * eslen
                seg_points_list.insert(-2,separation_len-1)
            seg_points = np.array(seg_points_list)
            # print(seg_points)
        for p in seg_points:
            edge[p, (border-1+side), 1:] = 0  # turn to red

        img_t = np.concatenate((corner_1, edge, corner_2), axis=0)
        img = Image.fromarray(img_t)
        img.save(mode+'_seg/'+str(edge_len)+'_'+str(n)+'.png')

def get_seg_lengths(img_file):
    img = Image.open(img_file)
    img_t = np.array(img)
    coor = []
    for i in range(img_t.shape[0]):
        for j in range(img_t.shape[1]):
            if (img_t[i,j] != [0,0,0]).any() and (img_t[i,j] != [255,255,255]).any():
                coor.append([i,j])

    coor = np.array(coor)
    length = np.array([coor[i+1]-coor[i] for i in range(len(coor)-1)])
    length = length.flatten()
    lengths = [i for i in length if i != 0]
    lengths[0] += 1  # adjust the coor
    # print(lengths)
    cat_segs = len(lengths) if len(lengths) < 3 else 3  # category of segmentations
    # print(f"cat_segs={cat_segs}")
    return lengths, cat_segs

def kmeans_learn():
    X_train = []
    filepath = 'train_seg/'
    files = os.listdir(filepath)
    cat_seg_1_len_max = 0
    # cat_seg_2_len_min = cat_seg_1_len_max
    cat_seg_2_len_max = 0
    # cat_seg_3_len_min = cat_seg_2_len_max
    cat_seg_3_len_max = 0
    i = 0
    for file in files:
        lengths, cat_segs = get_seg_lengths(filepath+file)
        edge_length = sum(lengths)
        # print(lengths, cat_segs, edge_length)
        if cat_segs == 1:  # no seg
            cat_seg_1_len_max = max(cat_seg_1_len_max, edge_length)
        elif cat_segs == 2:  # one seg
            # cat_seg_2_len_min = cat_seg_1_len_max
            cat_seg_2_len_max =  max(cat_seg_2_len_max, edge_length)
        elif cat_segs == 3:  # two or more segs
            # cat_seg_3_len_min = cat_seg_2_len_max
            cat_seg_3_len_max =  max(cat_seg_3_len_max, edge_length)
            X_train += [[i] for i in lengths]

    cat_seg_2_len_min = cat_seg_1_len_max
    cat_seg_3_len_min = cat_seg_2_len_max
    print(cat_seg_1_len_max)
    print(cat_seg_2_len_min, cat_seg_2_len_max)
    print(cat_seg_3_len_min, cat_seg_3_len_max)
    print(X_train)
    for n_clusters in range(3,7):
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_jobs=6)
        kmeans.fit(X_train)
        print(kmeans.cluster_centers_)
        # print(type(kmeans.labels_))
        num_per_cluster = [0]*n_clusters
        for i in range(n_clusters):
            num_per_cluster[i] = (i == kmeans.labels_).sum()
        print(num_per_cluster)

        plt.clf()
        for i in range(n_clusters):
            plt.bar(kmeans.cluster_centers_[i], num_per_cluster[i], width=30)
        plt.xticks([np.around(i,0) for i in kmeans.cluster_centers_], fontsize=10)
        plt.yticks(fontsize=10)
        for x, y in zip(kmeans.cluster_centers_, num_per_cluster):
            plt.text(x + 0.005, y + 0.005, '%.f' % y, ha='center', va='bottom', fontsize=10)
        plt.xlabel('Cluster centers', fontsize=10)
        plt.ylabel('Number of instances per cluster', fontsize=10)
        plt.title(f'Number of clusters = {n_clusters}', fontsize=15)
        plt.tight_layout()
        plt.savefig(f'cluster_{n_clusters}.png') 

def infer_corner(svm_model_name, mlp_model_name):
    files = os.listdir('inference_corner')
    clf_svm = joblib.load(svm_model_name)
    clf_mlp = joblib.load(mlp_model_name)
    X = []
    for file in files:
        print(file)
        img = Image.open('inference_corner/'+file)
        X.append(np.array(img).flatten())
    svm_prediction = clf_svm.predict(X)
    mlp_prediction = clf_mlp.predict(X)
    print(svm_prediction)
    print(mlp_prediction)

def main():
    svm_learn()
    mlp_learn()
    infer_corner(SVM_MODEL_NAME, MLP_MODEL_NAME)

if __name__ == '__main__':
    main()