import cv2
import os
import numpy as np
import random
import shutil
import sys
def changelocate(cx, cy, r, p):
    def cgdeta(g):
        x = np.cos(g)
        y = np.sin(g)
        return [x, y]

    points = []
    dx = 90
    angle = []
    for n in range(10):
        angle.append(dx)
        dx -= 36

    for e in angle:
        x = r * np.cos(e) + cx
        y = cy - r * np.sin(e)
        points.append([int(x), int(y)])
    return points

mode='train'

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

    input = cv2.imread(path + imgname)
    dst = cv2.pyrMeanShiftFiltering(input, 10, 100)
    cimage = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 80, param1=100, param2=20, minRadius=80, maxRadius=0)
    circles = np.uint16(np.around(circles))  # 把类型换成整数
    r_1 = circles[0, 0, 2]
    c_x = circles[0, 0, 0]
    c_y = circles[0, 0, 1]

    points = changelocate(c_x, c_y, r_1 * 0.65, 27)
    font = cv2.FONT_HERSHEY_SIMPLEX
    weight = 1
    order = 0
    boxes = []
    numl = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    pic_path = os.path.join(os.getcwd(), trainpath + '/' + imgname)
    pic_path=pic_path.replace("\\","/")
    print('picpath',pic_path)

    with open(txtpath, 'a') as f_w:
        f_w.write(pic_path)
        for x, y in points:
            x = x - 20
            cv2.putText(input, str(order), (x, y), font, weight, (255, 255, 255), 2)
            box1 = [str(x), str(int(y - 25 * weight)), str(int(x + 20 * weight)),str(y+5),str(order)]
            box1= ' ' + ','.join(box1)
            f_w.write(box1)
            # cv2.rectangle(input, (x, int(y - 25 * weight)), (int(x + 20 * weight), y + 5), (234,43,223), 1)
            order += 1
            nx = x + 26
            cv2.putText(input, '0', (nx, y), font, weight, (255, 255, 255), 2)
            # cv2.rectangle(input, (nx, int(y - 25 * weight)), (int(nx + 20 * weight), y + 5), (234, 43, 223), 1)
            box2 = [str(nx), str(int(y - 25 * weight)), str(int(nx + 20 * weight)), str(y + 5),'0']
            box2 = ' ' + ','.join(box2)
            cv2.imwrite(pic_path, input)
            f_w.write(box2)
        f_w.write('\n')
print('read_done')
