# coding=utf-8
# 在0-2*pi的区间上生成100个点作为输入数据
import numpy as np
from matplotlib import pyplot as pt
from PIL import Image

def PCA(data, k):
    '''
    PCA算法
    :param data:需要处理的数据
    :param k:需要截取的维度
    :return:low_data降维以后投影的数据,pri_data在新的特征向量下的数据
    '''
    mean_col = np.mean(data, axis=0)
    centre_data = data - mean_col # 得出中心化的数据
    cov = np.cov(centre_data.transpose())  # 求出协方差,这里是把行当做变量所以要转置
    value,vector = np.linalg.eig(cov)  # 提取出特征值,特征向量
    value_sort = np.argsort(value)
    # len = len(value_sort[0])
    value_index = value_sort[-1:-(k+1):-1]
    # print("sort",value_sort)
    # print("value",value)
    # print("vector",vector)
    cut_vector = vector[:,value_index]  # 选取需要的k个特征
    low_data = np.dot(centre_data,cut_vector)  # 得出低维空间的数据
    # print("low大小",low_data.shape)
    # print("cut_vector大小", np.array(cut_vector).shape)
    # print("cen_data大小", centre_data.shape)
    pri_data = np.dot(low_data,np.transpose(cut_vector))+mean_col  # 降维之后的样本
    return low_data, pri_data


def vector_to_picture(data):
    '''

    :param data: 降维以后的图像的数组
    :return:展示降维以后的图像的效果
    '''
    new_im = Image.fromarray(255*data)   # 把数据重新组织变成图像
    new_im.show() # 显示图片

def img_to_matrix(img_name):
    '''
    把图片变成矩阵输出
    :param img_name: 图片的名字
    :return: 能表示图片的矩阵
    '''
    im = Image.open(img_name)
    # im.show()
    width, height = im.size
    # print("图片的宽和高是：", width, height)
    im = im.convert("L")
    data = im.getdata()
    # data = np.matrix(data)  # 把读取到的图像的数据变成矩阵
    data = np.matrix(data, dtype='float') / 255.0
    print(data)
    new_data = np.reshape(data, (height, width))  # 把读取的信息变成数组的形式
    print(new_data)
    return new_data

def main():
    img_name = "2.jpg"
    k = 40  # 保留的维度
    data = img_to_matrix(img_name)
    low_data, pri_data = PCA(data, k)
    vector_to_picture(np.array(pri_data.real))

if __name__ == '__main__':
    main()
