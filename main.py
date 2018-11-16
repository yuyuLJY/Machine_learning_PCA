# coding=utf-8
# 在0-2*pi的区间上生成100个点作为输入数据
import numpy as np
from PIL import Image
import struct

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
    value_index = value_sort[-1:-(k+1):-1]
    cut_vector = vector[:,value_index]  # 选取需要的k个特征
    low_data = np.dot(centre_data,cut_vector)  # 得出低维空间的数据
    pri_data = np.dot(low_data,np.transpose(cut_vector))+mean_col  # 降维之后的样本
    return low_data, pri_data


def vector_to_picture(data):
    '''

    :param data: 降维以后的图像的数组
    :return:展示降维以后的图像的效果
    '''
    data = np.array(data).reshape(28, 28)
    data = np.array(data).real
    new_im = Image.fromarray(data)   # 把数据重新组织变成图像
    new_im.show() # 显示图片
'''
def img_to_matrix(img_name):
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
'''

def get_label():
    f1=open("train-labels.idx1-ubyte",'rb')
    buf1=f1.read()
    f1.close()
    index=0
    magic,num=struct.unpack_from(">II",buf1,0)
    index+=struct.calcsize('>II')
    labs=[]
    labs=struct.unpack_from('>'+str(num)+'B',buf1,index)
    return labs #返回训练标签。之前没有单独解析出来保存在文本文件中，因为解析标签比较简单。


def getImages():  # 处理训练图片
    images_path = "train-images.idx3-ubyte"
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    return images

def PSNR_test(images,pri_data):
    print("计算SNR")
    row,col = np.array(images).shape
    print(row,col)
    sum =0

    up = 0
    down = 0
    pri_data = np.array(pri_data).real
    images = np.array(images).real
    for i in range(10):
        for j in range(10):
            print(pri_data[i,j])
            up += np.power(pri_data[i,j],2)
            down += np.power((pri_data[i,j]-images[i,j]),2)
    '''
    for i in range(500):
        for j in range(col):
            sum += (images[i][j]-pri_data[i][j])**2
            print(sum)

    print("end")
    a = np.sum((images-pri_data))/784
    P = 10*np.log10(255**2/a)
    print(np.array(P).real)
    '''
    
    SNR = 10*np.log10(up/down)
    print(SNR)

if __name__ == '__main__':
    k = 8
    path = "data"
    labels = get_label()
    # print(labels)
    images = getImages()  # 获取数据
    print(len(images[0]))
    # print(images)
    # 已经读取完毕
    low_data, pri_data = PCA(images,2)
    # print(len(pri_data[0])) #784
    # print(test)
    # vector_to_picture(pri_data[0])
    print(images)
    print(len(pri_data[0]))
    print(np.shape(pri_data))
    vector_to_picture(pri_data[0])
    PSNR_test(images,pri_data)

