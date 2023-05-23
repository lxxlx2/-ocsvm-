import cv2
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import numpy as np
import pickle

from sklearn.svm import OneClassSVM
from feature_extract import *
import numpy as np
from keras.applications import VGG16
from keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
import shutil


def T(svm, test_dir_true, test_dir_false, test_result_dir, nu, gamma):
    print(nu, gamma)
    true_value = test(svm, test_dir_true, test_result_dir, nu, gamma) / 25
    false_value = (25 - test(svm, test_dir_false, test_result_dir, nu, gamma)) / 25
    print((true_value + false_value) / 2 * 100.)
    return (true_value + false_value) / 2 * 100.


def test(svm, test_dir, test_result_dir, nu, gamma):
    true_num = 0  # 被分类为正例的图片数量(有火灾)
    false_num = 0  # 被分类为负例的图片数量(无火灾)

    # 若test_result已存在，则删除并重新创建该文件夹
    if os.path.exists(test_result_dir):
        shutil.rmtree(test_result_dir)

    # 检查父文件夹是否存在
    if not os.path.exists(os.path.dirname(test_result_dir)):
        # 如果父文件夹不存在，则创建它
        os.makedirs(os.path.dirname(test_result_dir))

    # 创建子文件夹
    os.mkdir(test_result_dir)
    os.mkdir(os.path.join(test_result_dir, "true"))
    os.mkdir(os.path.join(test_result_dir, "false"))

    for file_name in os.listdir(test_dir):  # 遍历test_dir
        file_path = os.path.join(test_dir, file_name)  # 文件路径
        images = cv2.imread(file_path)  # 读取图片
        if images is None:
            print("读取图片" + file_path + "失败")
            continue

        features = extract_features(file_path, model)
        if np.sum(features) > 0:
            features = features / np.sum(features)
        # features = features[:, np.newaxis]  # 增加一维,变为二维
        # scaler = StandardScaler()  # 标准化
        # features_scaled = scaler.fit_transform(features)
        # features = np.squeeze(features_scaled)  # 还原一维

        rgb = GetFeature(images)  # 提取图片的特征向量
        features = np.concatenate([rgb, features], axis=0)

        prediction = svm.predict([features])  # 使用SVM预测类别

        if prediction[0] == -1:
            # print("图片{}被分类为异常值".format(file_name))
            shutil.copyfile(file_path, "output/test_result/false/" + file_name)  # 复制文件
            false_num = false_num + 1
        else:
            # print("图片{}被分类为正常值".format(file_name))
            shutil.copyfile(file_path, "output/test_result/true/" + file_name)  # 复制文件
            true_num = true_num + 1

    print("共有{}个正常值，{}个异常值。\n正常值比例：{:.1f}%，异常值比例：{:.1f}%".format(true_num, false_num,
                                                                 100. * true_num / (true_num + false_num),
                                                                 100. * false_num / (true_num + false_num)))
    return true_num


def extract_features(image_path, model):
    img = image.load_img(image_path, target_size=(32, 32))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    features = np.array(features).flatten()
    # mean = np.mean(features)
    # std = np.std(features)
    # features = (features - mean) / std
    return features


if __name__ == '__main__':
    train_dir = "data/images/train0"
    test_dir_true = "data/Test_Data/Fire"
    test_dir_false = "data/Test_Data/Non_Fire"
    test_result_dir = "output/test_result"

    # 加载训练集图片
    train_data = []

    # features = GetFeature(image) # 提取图片的特征向量
    # train_data.append(features)

    # 提取特征模型
    model = VGG16(weights='imagenet', include_top=False)

    for file_name in os.listdir(train_dir):  # 遍历test_dir
        file_path = os.path.join(train_dir, file_name)  # 文件路径
        images = cv2.imread(file_path)  # 读取图片
        if images is None:
            print("读取图片" + file_path + "失败")
            continue

        features = extract_features(file_path, model)
        if np.sum(features) > 0:
            features = features / np.sum(features)
        # features = features[:, np.newaxis]  # 增加一维,变为二维
        # scaler = StandardScaler()  # 标准化
        # features_scaled = scaler.fit_transform(features)
        # features = np.squeeze(features_scaled)  # 还原一维

        rgb = GetFeature(images)  # 提取图片的特征向量
        features = np.concatenate([rgb, features], axis=0)
        train_data.append(features)

    # features = [train_data[i] for i in range(train_data.shape[0])]

    # 网格搜索nu和gamma参数
    # nu_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    # gamma_values = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    nu_values = [0.15]
    gamma_values = [0.5]
    best_res = -1

    for nu in nu_values:
        for gamma in gamma_values:
            svm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
            svm.fit(train_data)

            # 10折交叉验证
            # scores = cross_val_score(svm, train_data, scoring='f1', cv=5)
            # score = np.mean(scores)
            res = T(svm, test_dir_true, test_dir_false, test_result_dir, nu, gamma)
            if res > best_res:
                best_res = res
                with open('model.pkl', 'wb') as f:
                    pickle.dump(svm, f)
            # test(svm, test_dir, nu, gamma)
            # if score > best_score:
            #     best_nu = nu
            #     best_gamma = gamma
            #     best_score = score
            #     print("score")

    # 使用最佳参数重新训练模型
    # svm = OneClassSVM(kernel='rbf', nu=best_nu, gamma=best_gamma)
    # svm.fit(train_data)
    #
    # # 将模型保存到磁盘
    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(svm, f)

    # 使用测试集图片进行测试
    # res = T(svm, test_dir_true, test_dir_false, test_result_dir, nu, gamma)