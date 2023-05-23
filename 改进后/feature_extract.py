import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
# from sklearn.feature_extraction.image import hog
from skimage import feature
import torchvision.models as models
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
from sklearn.preprocessing import StandardScaler

def GetFeature(image):
    # 加载图像并预处理
    img_path = 'image.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # 加载VGG16模型,不包含分类器
    model = VGG16(weights='imagenet', include_top=False)

    # 提取特征
    features = model.predict(x)

    return features


def GetFeature(image):
    rgb_features = GetRgbFeature(image)

    # 归一化特征向量
    if np.sum(rgb_features) > 0:
        rgb_features = rgb_features / np.sum(rgb_features)
    features = rgb_features

    # rgb_features = rgb_features[:, np.newaxis]  # 增加一维,变为二维
    # scaler = StandardScaler()  # 标准化
    # features_scaled = scaler.fit_transform(rgb_features)
    # features = np.squeeze(features_scaled)  # 还原一维

    return features

def GetRgbFeature(image):
    b, g, r = cv2.split(image)

    # 计算颜色直方图
    hist_r = cv2.calcHist([r], [0], None, (256,), [0, 256])
    hist_g = cv2.calcHist([g], [0], None, (256,), [0, 256])
    hist_b = cv2.calcHist([b], [0], None, (256,), [0, 256])

    features = np.concatenate([hist_r, hist_g, hist_b], axis=-1).flatten()

    # model = VGG16(weights='imagenet', include_top=False)
    # img = cv2.resize(image, (224, 224))
    # x = image.img_to_array(img)  # 转化为浮点型
    # x = np.expand_dims(x, axis=0)  # 转化为张量size为(1, 224, 224, 3)
    # x = preprocess_input(x)
    #
    # # 預測，取得features，維度為 (1,1000)
    # features = model.predict(x)

    return features

def GetHogFeature(image, cell_size=(8, 8), block_size=(3, 3), nbins=7):
    # 计算梯度和方向直方图
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(nbins * ang / (2 * np.pi))

    # 划分cell并计算cell的直方图
    cell_size_x, cell_size_y = cell_size
    bin_cells = bins[:cell_size_x * (image.shape[0] // cell_size_y), :cell_size_y * (image.shape[1] // cell_size_x)].reshape(
        image.shape[0] // cell_size_y, cell_size_y, image.shape[1] // cell_size_x, cell_size_x)
    mag_cells = mag[:cell_size_x * (image.shape[0] // cell_size_y), :cell_size_y * (image.shape[1] // cell_size_x)].reshape(
        image.shape[0] // cell_size_y, cell_size_y, image.shape[1] // cell_size_x, cell_size_x)

    hist_cells = np.zeros((image.shape[0] // cell_size_y, image.shape[1] // cell_size_x, nbins), dtype=np.float32)
    for i in range(nbins):
        hist_cells[..., i] = np.sum(mag_cells * (bin_cells == i), axis=(1, 3))

    # 划分block并归一化
    block_size_x, block_size_y = block_size
    norm_cells = np.zeros((image.shape[0] // cell_size_y - block_size_y + 1, image.shape[1] // cell_size_x - block_size_x + 1, block_size_y * block_size_x * nbins), dtype=np.float32)
    for i in range(norm_cells.shape[0]):
        for j in range(norm_cells.shape[1]):
            block = hist_cells[i:i + block_size_y, j:j + block_size_x, :].reshape(-1)
            norm_cells[i, j, :] = block / np.sqrt(np.sum(block ** 2) + 1e-5)

    # 将所有块的特征向量连接起来
    hog_feat = norm_cells.reshape(-1)

    return hog_feat

def extract_features(image_path, model):
    img = image.load_img(image_path, target_size=(32, 32))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    features = np.array(features).flatten()

    # features = features[:, np.newaxis]  # 增加一维,变为二维
    #
    # # 标准化
    # scaler = StandardScaler()
    # features_scaled = scaler.fit_transform(features)
    # # 还原一维
    # features = np.squeeze(features_scaled)

    # mean = np.mean(features)
    # std = np.std(features)
    # features = (features - mean) / std
    return features
    