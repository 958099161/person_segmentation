import cv2
import os
from PIL import  Image
from tqdm import tqdm
import numpy as np
#图片路径
def change_label():
    img_str='E:\Oppo_Game\pytorch-semantic-segmentation-master/data/dataset/gtFine'
    out_str='E:\Oppo_Game\pytorch-semantic-segmentation-master/data/dataset/label255'
    list_img=os.listdir(img_str)
    for img_name in tqdm(list_img):
        name_path = os.path.join(img_str, img_name)
        img2 = cv2.imread(name_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('', img2)
    # cv2.waitKey(0)
    #保存图片
        Image.fromarray(img2).save(os.path.join(out_str, img_name))
def show255():
    # str_path ='E:\Oppo_Game\seg_01'
    # out_str = 'E:\Oppo_Game\seg_out/00086-profile.jpg'
    img_in='E:\Oppo_Game\pytorch-semantic-segmentation-master/data/dataset/gtFine'
    img_out='E:\Oppo_Game\pytorch-semantic-segmentation-master/data/dataset/label255'
    # out_seg ='E:\Oppo_Game\seg_out'
    # out_path ='E:\Oppo_Game\seg_01'
    list_img=os.listdir(img_in)
    for img_name in tqdm(list_img):
        name_path = os.path.join(img_in, img_name)
        a=Image.open(name_path)
    # a.show()
        b=np.array(a)
        b[b >= 1] = 255
        c=Image.fromarray(b)
        # c.show()
        c.save(os.path.join(img_out, img_name))
    print(b)
    print("end")
def test():
    img_in='E:\Oppo_Game\pytorch-semantic-segmentation-master\data\dataset\label0/00001-profile.jpg'
    a=Image.open(img_in)
    # a.show()
    b=np.array(a)
    b[b>=1]=255
    b[b<0.5]=0
    # b[b<50]=1
    c=Image.fromarray(b)
    c.show()
    print('hello')

# str_path ='E:\Oppo_Game\seg_01'
# out_str = 'E:\Oppo_Game\seg_out/00086-profile.jpg'
def change_01():
    img_in='E:\Oppo_Game\pytorch-semantic-segmentation-master\data\dataset\label0'

    img_out='E:\Oppo_Game\pytorch-semantic-segmentation-master/data/dataset/gtFine'

    # out_seg ='E:\Oppo_Game\seg_out'

    # out_path ='E:\Oppo_Game\seg_01'
    list_img=os.listdir(img_in)
    for img_name in tqdm(list_img):
        name_path = os.path.join(img_in, img_name)

        a=Image.open(name_path)
    # a.show()
        b=np.array(a)
        b[b <10] = 0
        b[b >= 10] = 1
        # b[b < 200] = 0
        c=Image.fromarray(b)
        # c.show()
        c.save(os.path.join(img_out, img_name.split('.')[1]+'.ipg'))
    print(b)
    print("end")



def make_label():
    img_in = 'E:\lable255'
    img_out = 'E:\lable0'

    # out_seg ='E:\Oppo_Game\seg_out'
    # out_path ='E:\Oppo_Game\seg_01'
    list_img = os.listdir(img_in)
    for img_name in tqdm(list_img):
        name_path = os.path.join(img_in, img_name)

        in_img = cv2.imread(name_path)
        b = in_img[:,:,0]


        # b = np.array(in_img)
        b[b >= 1] = 100
        b[b < 1] = 255
        b[b <150] = 0

        cv2.imwrite(os.path.join(img_out, img_name.split('.')[0] + '-profile.jpg'),b)
        # c = Image.fromarray(b)
        # c.show()
        # c.save(os.path.join(img_out, img_name.split('.')[0] + '.jpg'))
        # c.save(os.path.join(img_out, img_name))
    # print(b)
    print("end")

if __name__=='__main__':
    make_label()