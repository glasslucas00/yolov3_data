from PIL import Image, ImageFilter
import matplotlib.pyplot as  plt
from pylab import *
import numpy as np
# im = Image.open('test/01.jpg')
import cv2
"""   
https://pillow.readthedocs.io/en/3.3.x/reference/ImageFilter.html?highlight=ImageFilter.BLUR
滤镜名称	方法
模糊滤镜	ImageFilter.BLUR
铅笔轮廓滤镜	ImageFilter.CONTOUR
浮雕滤镜	ImageFilter.EMBOSS
边缘凸显滤镜	ImageFilter.EDGE_ENHANCE
边缘凸显滤镜（加强）	ImageFilter.EDGE_ENHANCE_MORE
只保留滤镜	ImageFilter.FIND_EDGES
锐化滤镜	ImageFilter.SHARPEN
平滑滤镜	ImageFilter.SMOOTH
平滑滤镜（加强）	ImageFilter.SMOOTH_MORE
高斯模糊滤镜 ImageFilter.GaussianBlur（radius = 2 ）
锐化蒙版滤镜  ImageFilter.UnsharpMask（radius = 2，percent = 150，threshold = 3 ）

"""



def filter_image(cvimg,cvpath):
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

    subplot(2, 3, 1)
    plt.imshow(im1)
    subplot(2, 3, 2)
    plt.imshow(im2)
    subplot(2, 3, 3)
    plt.imshow(im3)
    subplot(2, 3, 4)
    plt.imshow(im4)
    subplot(2, 3, 5)
    plt.imshow(im5)
    subplot(2, 3, 6)
    plt.imshow(im6)
    plt.show()


# if __name__ =='__main__':
#     filter_image(img,aa)



