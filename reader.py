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
    print(filepath,'\n',rootpath)
    img=Image.fromarray(cvimg)
    im1 = img.filter(ImageFilter.GaussianBlur(1))
    im2 = img.filter(ImageFilter.UnsharpMask)
    im3 = img.filter(ImageFilter.SHARPEN)
    im4 = img.filter(ImageFilter.SMOOTH)
    im5 = img.filter(ImageFilter.SMOOTH_MORE)
    im6 = img.filter(ImageFilter.EDGE_ENHANCE)
    im1.save(rootpath+'im1'+filepath)
    im2.save(rootpath + 'im2' + filepath)
    im3.save(rootpath + 'im3' + filepath)
    im4.save(rootpath + 'im4' + filepath)
    im5.save(rootpath + 'im5' + filepath)
    im6.save(rootpath + 'im6' + filepath)
    with open(txtpath, 'a') as f_w:
        f_w.write(rootpath+'im1'+filepath+strbox+'\n')
        f_w.write(rootpath + 'im2' + filepath + strbox + '\n')
        f_w.write(rootpath + 'im3' + filepath + strbox + '\n')
        f_w.write(rootpath + 'im4' + filepath + strbox + '\n')
        f_w.write(rootpath + 'im5' + filepath + strbox + '\n')
        f_w.write(rootpath + 'im6' + filepath + strbox + '\n')

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
            dst = cv2.pyrMeanShiftFiltering(input, 10, 100)
            cimage = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 80, param1=100, param2=20, minRadius=80, maxRadius=0)
            circles = np.uint16(np.around(circles))  # 把类型换成整数
            r_1 = circles[0, 0, 2]
            c_x = circles[0, 0, 0]
            c_y = circles[0, 0, 1]

            points = changelocate(c_x, c_y, r_1 * 0.65)
            fonts=[cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_PLAIN,cv2.FONT_HERSHEY_DUPLEX]
            font = 0
            ss=[0.9,1,1.1]
            weight = ss[randint(0,3)]
            order = 0
            boxes = []
            numl = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
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
                    nx = x + 23
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
                    # cv2.waitKey(500)
                f_w.write('\n')
            filter_image(input, pic_path, f_s)
        except:
            pass
    print('read_done')
markdata('test')