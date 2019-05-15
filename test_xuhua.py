import argparse
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import torch.optim
# import U_net
from utils.eval_segm import mean_IU
from utils.crf import DenseCRF
from networks.erfnet import ERFNet

parser = argparse.ArgumentParser(description='Person Segmentation')

parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--model_dir', default='./save_models/erfnet_3_11_non_big300.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')  # default='./save_models/erfnet3_8_5.pth'

parser.add_argument('-r', '--data_dir', default=r'E:\game\pytorch-semantic-segmentation_person\test_miou\real',
                    metavar='DIR',
                    help='path to image')
parser.add_argument('-o', '--output', default='./test_miou/seg_crf', metavar='DIR',
                    help='path to output')
args = parser.parse_args()


def test_mIou():
    list_path = os.listdir('./test_miou/label')
    out_iou = []
    step = 0
    for name_img in list_path:
        try:
            step = step + 1
            in_img = cv2.imread('./test_miou/label/' + name_img)
            in_img = in_img[:, :, 0]

            img = cv2.imread('./test_miou/seg_crf/' + name_img.split('-')[0] + '.png')
            img2 = img[:, :, 0]
            out = mean_IU(in_img, img2)
            out_iou.append(out)
        except:
            print(name_img)
    # print(sum(out))
    print(out_iou)
    print(sum(out_iou) / len(out_iou))


def inference():
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    model = ERFNet(2)
    # model = UNet(2)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model_dir))
    model.eval()
    list_images = os.listdir(args.data_dir)
    with torch.no_grad():
        for name in tqdm(list_images):
            img_name = os.path.join(args.data_dir, name)
            imagepro = cv2.imread(img_name)  # 宽×高×通道
            gauss = cv2.GaussianBlur(imagepro, ksize=(9, 9), sigmaX=0, sigmaY=0)  # 高斯模糊后图像
            h, w, _ = imagepro.shape
            image = (imagepro / 256 - 0.5).transpose((2, 0, 1))  # 3  h   wERFNet
            image = torch.from_numpy(image).float().unsqueeze(0).cuda()  # 3  h   w    double

            out_img = model(image)

            # ++++++++++++++++++++++++++++++++++++++加crf
            try:
                # #
                # postprocessor = DenseCRF(
                #     iter_max=2,
                #     pos_xy_std=1,
                #     pos_w=3,
                #     bi_xy_std=67,
                #     bi_rgb_std=3,
                #     bi_w=4,
                # )
                # probs = F.softmax(out_img, dim=1)
                # probs = probs.data.cpu().numpy()[0]
                # # Refine the prob map with CRF
                # _,out_h,out_w=probs.shape
                # if out_h!=h or out_w!=w:
                #     begin_h = (out_h - h) // 2
                #     end_h = begin_h + h
                #
                #     begin_w = (out_w - w) // 2
                #     end_w = begin_w + w
                #     probs=probs[:,begin_h:end_h,begin_w:end_w]
                # if postprocessor and imagepro is not None:
                #     probs = postprocessor(imagepro, probs)
                # probs = torch.from_numpy(probs)
                # output_np = torch.max(probs, 0)[1]
                # np_out = output_np.cpu().numpy()  # 1,480  ,360
                # np_out = np_out.squeeze()
                # np_out[np_out >= 1] = 255
                # np_out[np_out < 1] = 0
                # img_ = Image.fromarray(np.uint8(np_out))


                # ++++++++++++++++++++++++++++++++++++++没有加crf

                out_img = torch.max(out_img, 1)[1]
                out_img = out_img.cpu().numpy().squeeze()  # 1,480  ,360
                # out_img = out_img
                # out_img = out_img.squeeze()
                out_h, out_w = out_img.shape
                out_img[out_img >= 0.5] = 255
                out_img[out_img < 0.5] = 0
                if out_h != h or out_w != w:
                    # img_.resize((h, w))
                    begin_h = (out_h - h) // 2
                    end_h = begin_h + h

                    begin_w = (out_w - w) // 2
                    end_w = begin_w + w

                    out_img = out_img[begin_h:end_h, begin_w:end_w]
                img_ = Image.fromarray(np.uint8(out_img))

                in_img = np.array(img_)
                # print(in_img.shape)
                in_img[in_img >= 0.5] = 1
                in_img[in_img < 0.5] = 0  # 0，1矩阵

                person_img = np.zeros((h, w, 3))
                person_img[:, :, 0] = in_img * imagepro[:, :, 0]
                person_img[:, :, 1] = in_img * imagepro[:, :, 1]
                person_img[:, :, 2] = in_img * imagepro[:, :, 2]
                # cv2.imwrite(args.output + '//' + name.split('.')[0] + '_person.png', person_img)
                # person_img[:,:,0][person_img[:,:,0] != 0] = gauss[:,:,0]
                # person_img[:, :, 1][person_img[:, :, 1] != 0] = gauss[:, :, 1]
                # person_img[:, :, 2][person_img[:, :, 2] != 0] = gauss[:, :, 2]
                for i in range(h):              # 模糊部分和人高清部分拼接
                    for j in range(w):
                        if in_img[i][j]==0:
                            person_img[i,j,:] = gauss[i,j,:]
                cv2.imwrite(args.output + '//' + name.split('.')[0] + 'guass.png',person_img)
                img_.save(args.output + '//' + name.split('.')[0] + '.png')
            except:
                print(name)


if __name__ == '__main__':
    inference()
    # test_mIou()
