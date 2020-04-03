import os
import numpy as np
import cv2
import random
Angles = [10, -5,10,-30,40,60]
Trans_Select_Imgs = 6
def read_imgs(imgs_path):
    imgs_name = os.listdir(imgs_path)
    imgs = []
    for img_name in imgs_name:
        img_path = os.path.join(imgs_path, img_name)
        img = cv2.imread(img_path)
        cv2.resize(img,(416,416))
        imgs.append(img)
    return imgs

# 旋转
def rotate_img(img, angle):
    (height, width) = img.shape[:2]
    center = (height // 2, width // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 旋转图像
    rotate_img = cv2.warpAffine(img, matrix, (width, height))
    return rotate_img


def get_rotate_imgs(imgs, rotate_img_path):
    ii = 0
    for idx_img in range(len(imgs)):
        for angle in Angles:
            r_img = rotate_img(imgs[idx_img], angle)
            rotate_img_name = str(idx_img)+str(ii) + '.jpg'
            ii+=1
            cv2.imwrite(rotate_img_path + rotate_img_name, r_img)


# 平移
def translate_img(img, x_shift, y_shift):
    (height, width) = img.shape[:2]
    # 平移矩阵(浮点数类型)  x_shift +右移 -左移  y_shift -上移 +下移
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    # 平移图像
    trans_img = cv2.warpAffine(img, matrix, (width, height))
    return trans_img


def get_trans_imgs(imgs, trans_img_path):
    imgs_num = len(imgs)
    np.random.seed(2)

    random_imgs = np.random.randint(0, imgs_num, Trans_Select_Imgs)
    for img_idx in random_imgs:
        # 获得随机平移坐标
        x_shift = np.random.randint(-30,30, 1)
        y_shift = np.random.randint(-30,30, 1)
        trans_img = translate_img(imgs[img_idx], x_shift, y_shift)
        # 保存平移图片和平移坐标
        trans_img_name = str(img_idx) + '_' + str(x_shift[0]) + '_' + str(y_shift[0]) + '.jpg'
        # cv2.imshow('r',trans_img)
        # cv2.waitKey(1)
        cv2.imwrite(trans_img_path + trans_img_name, trans_img)
#批量二值化
def bin_deal(imgpath):
    paths=os.listdir(imgpath)
    for path in paths:
        path=os.path.join(imgpath,path)
        print(path)
        img=cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 先转换为灰度图才能够使用图像阈值化
        ret, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # cv2.imshow('s',th1)
        # cv2.waitKey(1)
        cv2.imwrite(path,th1)
def sp_noise(imgs,imgpath):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    probs = [0.001,0.005,0.007,0.01]
    i=100
    for image in imgs:
        prob = random.choice(probs)
        output = np.zeros(image.shape, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        i+=1
        # cv2.imshow('rr', output)
        cv2.imwrite(imgpath+str(i)+'a.jpg',output)
#批量重命名
def rename_pic(path):
    filelsit=os.listdir(path)
    print(len(filelsit))
    # currentpath = os.getcwd()
    # os.chdir(path)
    num=1000
    for name in filelsit:
        name=os.path.join(path,name)
        n_name=os.path.join(path,str(num)+'.jpg')
        os.rename(name, n_name)
        num+=1

if __name__ == "__main__":
    path='./pic/'
    imgs = read_imgs(path)
    sp_noise(imgs, path)
    # print('========图像旋转==========')
    # get_rotate_imgs(imgs, path)
    # print('========图像平移==========')
    # get_trans_imgs(imgs,path)
    # print('========图像重命名==========')
    # rename_pic(path)
    print('***********done************')