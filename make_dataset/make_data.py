import os
import cv2
from tqdm import tqdm
import numpy as np
def mk_img():
# f = open("foo.txt")
    list_map={}
    with open('person_trainval.txt') as f:
        line = f.readlines()
        for i in line:
            i_out=i .split()
            name =i_out[0]
            label =i_out[1]
            list_map[name]=label
            # print(i)

    list_img =os.listdir('F:\data_set\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationClass')
    for img_name in list_img:
        name =img_name.split('.')[0]
        if name in list_map:
            if list_map[name]=='1':
                img =  cv2.imread('F:\data_set\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationClass/'+name+'.png')
                cv2.imwrite('E:\label/'+name+'.png', img)


def mkjpg():
    list_name =[]
    list_img = os.listdir('E:\dataset_for_ubantu\icome_task2_data\clean_images\profiles')
    for img_name in list_img:
        name_path =img_name.split('-')[0]+'.jpg'
        img = cv2.imread( 'E:\dataset_for_ubantu\icome_task2_data\clean_images\images/' + name_path)
        cv2.imwrite('E:\dataset_for_ubantu\icome_task2_data\clean_images\img/' + name_path ,img)

    # # list_img =os.listdir('F:\data_set\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages')
    # # for img_name in list_img:
    # #     name =img_name.split('.')[0]
    # img = cv2.imread( 'F:\data_set\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationClass/' + name + '.png')
    # cv2.imwrite('E:\label/' + name + '.png', img)


def make_labels():
    img_in = 'E:\lable255'
    img_out = 'E:\llabel_one'
    list_img = os.listdir(img_in)
    for img_name in tqdm(list_img):
        name_path = os.path.join(img_in, img_name)
        in_img = cv2.imread(name_path)
        b = in_img[:,:,0]
        # b = np.array(in_img)
        b[b >= 10] = 100
        b[b < 10] = 255
        b[b <150] = 0
        cv2.imwrite('E:\label_one/'+img_name, b)
    print("end")

def make_label():
    img_in = 'E:\label'
    img_out = 'E:\lable255'

    # out_seg ='E:\Oppo_Game\seg_out'
    # out_path ='E:\Oppo_Game\seg_01'
    list_img = os.listdir(img_in)
    for img_name in tqdm(list_img):
        name_path = os.path.join(img_in, img_name)
        # out_arr=[]

        in_img = cv2.imread(name_path)
        b = in_img[:,:,0]
        c = in_img[:, :, 1]
        d = in_img[:, :, 2]
        # b = np.array(in_img)
        b[b != 128] = 0
        b[b == 128] = 255

        c[c != 128] = 0
        c[c==128]=255
        #
        d[d != 192] = 0
        d[d == 192] = 255


        out_arr = np.zeros((d.shape[0], d.shape[1]))
        for i in range(b.shape[0]):
            for j in range(c.shape[1]):
                if b[i][j]==255 and c[i][j]==255 and d[i][j]==255:
                    out_arr[i][j]=255

        # b[b <150] = 0

        cv2.imwrite(os.path.join(img_out, img_name.split('.')[0] + '-profile.jpg'),out_arr)
        # c = Image.fromarray(b)
        # c.show()
        # c.save(os.path.join(img_out, img_name.split('.')[0] + '.jpg'))
        # c.save(os.path.join(img_out, img_name))
    # print(b)
    print("end")

if __name__=='__main__':
    mkjpg()
