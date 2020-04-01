＃yolov3_data

基础图片在pic文件中，包含11张去数字仪表图像，再通过opencv生成数字得到数据集

generate.py 用于生成旋转，平移后的图像

reader.py 用于在仪表上添加数字，并将生成的位置box存储在train.txt/test.txt文件中

