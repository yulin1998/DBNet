# # # # # # # #
#@Author      : YuLin
#@Date        : 2022-07-07 20:02:50
#@LastEditors : YuLin
#@LastEditTime: 2022-07-08 16:37:53
#@Description : DBNet测试代码
# # # # # # # #

import gradio as gr
import cv2

import yaml
import torch
from models.DBNet import DBNet
from utils.tools import *
from utils.DB_postprocesss import *
import torchvision.transforms as transforms


def to_black(image):
    output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return output


def test_net(img):
    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream,Loader=yaml.FullLoader)
    model = DBNet(config)
    model_dict = torch.load(config['test']['checkpoints'])['state_dict']
    state = model.state_dict()
    for key in state.keys():
        if key in model_dict.keys():
            state[key] = model_dict[key]
    model.load_state_dict(state)
    model.eval()
    params = {'thresh':config['test']['thresh'],
              'box_thresh':config['test']['box_thresh'],
              'max_candidates':config['test']['max_candidates'],
              'is_poly':config['test']['is_poly'],
              'unclip_ratio':config['test']['unclip_ratio'],
              'min_size':config['test']['min_size']
              }
    dbprocess = DBPostProcess(params)

    img_ori = img.copy()
    img_name = 'test_img'
    img = resize_image(img,config['test']['short_side'])
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0)
    with torch.no_grad():
            out = model(img)
    scale = (img_ori.shape[1] * 1.0 / out.shape[3], img_ori.shape[0] * 1.0 / out.shape[2])
    bbox_batch,score_batch = dbprocess(out.cpu().numpy(),[scale])
    for bbox in bbox_batch[0]:
        bbox = bbox.reshape(-1, 2).astype(np.int32)
        # bbox = sort_coord(bbox)
        img_ori = cv2.drawContours(img_ori.copy(), [bbox], -1, (0, 255, 0), 2)
    # cv2.imwrite('./res.jpg', img_ori)
    return img_ori

interface = gr.Interface(fn=test_net, inputs="image", outputs="image")
interface.launch()