import cv2
import os
import numpy as np
import random
import shutil
import sys
from PIL import Image, ImageFilter
import matplotlib.pyplot as  plt
from pylab import *
# import skimage
# from skimage import util
global txtpath
def sp_noise(image):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    probs = [0.001,0.005,0.007]

    prob = np.random.choice(probs)
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
def changelocate(cx, cy, r):
    points = []
    dx=randint(80,95)
    # dx = random.randint(80,95)
    angle = []
    for n in range(10):
        angle.append(dx)
        dx -= 36

    for e in angle:
        x = r * np.cos(e) + cx
        y = cy - r * np.sin(e)
        points.append([int(x), int(y)])
    return points
def filter_image(cvimg,cvpath,strbox):
    path=cvpath.split('/')
    filepath=path[-1]
    rootpath =cvpath.replace(filepath,'')
    # print(filepath,'\n',rootpath)
    img=Image.fromarray(cvimg)
    im1 = img.filter(ImageFilter.GaussianBlur(1))
    im2 = img.filter(ImageFilter.UnsharpMask)
    im3 = img.filter(ImageFilter.SHARPEN)
    im4 = img.filter(ImageFilter.SMOOTH)
    im5 = img.filter(ImageFilter.SMOOTH_MORE)
    im6 = img.filter(ImageFilter.EDGE_ENHANCE)
    im7=sp_noise(cvimg)
    # print("********")
    # print(rootpath+'im1'+filepath)
    # print("********")
    im1.save(rootpath+'im1'+filepath)
    im2.save(rootpath + 'im2' + filepath)
    im3.save(rootpath + 'im3' + filepath)
    im4.save(rootpath + 'im4' + filepath)
    im5.save(rootpath + 'im5' + filepath)
    im6.save(rootpath + 'im6' + filepath)
    cv2.imwrite(rootpath + 'im7' + filepath,im7)
    with open(txtpath, 'a') as f_w:
        f_w.write(rootpath+'im1'+filepath+strbox+'\n')
        f_w.write(rootpath + 'im2' + filepath + strbox + '\n')
        f_w.write(rootpath + 'im3' + filepath + strbox + '\n')
        f_w.write(rootpath + 'im4' + filepath + strbox + '\n')
        f_w.write(rootpath + 'im5' + filepath + strbox + '\n')
        f_w.write(rootpath + 'im6' + filepath + strbox + '\n')
        f_w.write(rootpath + 'im7' + filepath + strbox + '\n')

    # subplot(2, 3, 1)
    # plt.imshow(im1)
    # subplot(2, 3, 2)
    # plt.imshow(im2)
    # subplot(2, 3, 3)
    # plt.imshow(im3)
    # subplot(2, 3, 4)
    # plt.imshow(im4)
    # subplot(2, 3, 5)
    # plt.imshow(im5)
    # subplot(2, 3, 6)
    # plt.imshow(im6)
    # plt.show()

def markdata(mode):
    global txtpath
    if mode=='train':
        path='pic_test/'
        trainpath='data/dataset/train'
        txtpath = 'data/dataset/yymnist_train.txt'
    elif mode=='test':
        path = 'pic_test/'
        trainpath='data/dataset/test'
        txtpath = 'data/dataset/yymnist_test.txt'
    else:
        print('错误：修改mode为train 或者 test')
        sys.exit()

    if os.path.exists(trainpath): shutil.rmtree(trainpath)
    os.mkdir(trainpath)


    imglist = os.listdir(path)
    for imgname in imglist:
        try:
            input = cv2.imread(path + imgname)
            nimg = input.copy()
            dst = cv2.pyrMeanShiftFiltering(input, 10, 100)
            cimage = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 80, param1=100, param2=20, minRadius=80, maxRadius=0)
            circles = np.uint16(np.around(circles))  # 把类型换成整数
            r_1 = circles[0, 0, 2]
            c_x = circles[0, 0, 0]
            c_y = circles[0, 0, 1]

            points = changelocate(c_x, c_y, r_1 * 0.65)
            fonts=[cv2.FONT_HERSHEY_PLAIN,cv2.FONT_HERSHEY_DUPLEX]
            font =cv2.FONT_HERSHEY_SIMPLEX
            ss=[0.9,1]
            weight = ss[randint(0,2)]
            order = 0
            # boxes = []
            # numl = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            pic_path = os.path.join(os.getcwd(), trainpath + '/' + imgname)
            pic_path=pic_path.replace("\\","/")
            print('picpath',pic_path)

            with open(txtpath, 'a') as f_w:
                f_s =''
                f_w.write(pic_path)
                for x, y in points:
                    x = x - 20
                    gray_value = randint(128, 255)
                    cv2.putText(input, str(order), (x, y), font, weight, (gray_value,gray_value,gray_value), 2)
                    box1 = [str(x), str(int(y - 25 * weight)), str(int(x + 20 * weight)),str(y+5),str(order)]
                    box1= ' ' + ','.join(box1)
                    # f_w.write(box1)
                    # cv2.rectangle(input, (x, int(y - 25 * weight)), (int(x + 20 * weight), y + 5), (234,43,223), 1)
                    order += 1
                    nx = x + 20
                    gray_value = randint(128, 255)
                    cv2.putText(input, '0', (nx, y), font, weight, (gray_value,gray_value,gray_value), 2)
                    # cv2.rectangle(input, (nx, int(y - 25 * weight)), (int(nx + 20 * weight), y + 5), (234, 43, 223), 1)
                    box2 = [str(nx), str(int(y - 25 * weight)), str(int(nx + 20 * weight)), str(y + 5),'0']
                    box2 = ' ' + ','.join(box2)
                    cv2.imwrite(pic_path, input)
                    f_w.write(box1+box2)
                    boxstr=(box1+box2)
                    f_s+=boxstr
                    # cv2.imshow('dd',input)
                    # cv2.waitKey(200)
                f_w.write('\n')
            filter_image(input, pic_path, f_s)

            pic_path = os.path.join(os.getcwd(), trainpath + '/cf' + imgname)
            pic_path = pic_path.replace("\\", "/")
            order = 0
            with open(txtpath, 'a') as f_w:
                f_s =''
                f_w.write(pic_path)
                font=5
                weight=1.2
                for x, y in points:
                    x = x - 20
                    gray_value = randint(128, 255)
                    cv2.putText(nimg, str(order), (x, y), font, weight, (gray_value,gray_value,gray_value), 2)
                    box1 = [str(x), str(int(y - 25 * weight)), str(int(x + 20 * weight)),str(y+5),str(order)]
                    box1= ' ' + ','.join(box1)
                    # f_w.write(box1)
                    # cv2.rectangle(nimg, (x, int(y - 20 * weight)), (int(x + 16 * weight), y + 5), (234,43,223), 1)
                    order += 1
                    nx = x + 18
                    gray_value = randint(128, 255)
                    cv2.putText(nimg, '0', (nx, y), font, weight, (gray_value,gray_value,gray_value), 2)
                    # cv2.rectangle(nimg, (nx, int(y - 20 * weight)), (int(nx + 16 * weight), y + 5), (234, 43, 223), 1)
                    box2 = [str(nx), str(int(y - 25 * weight)), str(int(nx + 20 * weight)), str(y + 5),'0']
                    box2 = ' ' + ','.join(box2)
                    cv2.imwrite(pic_path, nimg)
                    f_w.write(box1+box2)
                    boxstr=(box1+box2)
                    f_s+=boxstr
                    # cv2.imshow('d',nimg)
                    # cv2.waitKey(200)
                f_w.write('\n')
            filter_image(nimg, pic_path, f_s)
        except:
            pass

markdata('test')
print('********read_done**********')