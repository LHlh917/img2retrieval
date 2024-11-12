# # -*- coding: utf-8 -*-

# import numpy as np
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
# from keras.preprocessing import image
# from numpy import linalg as LA


# class VGGNet(object):
#     def __init__(self):
#         self.input_shape = (224, 224, 3)
#         self.weight = 'imagenet'
#         self.pooling = 'max'
#         self.model_vgg = VGG16(weights=self.weight,
#                                input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
#                                pooling=self.pooling,
#                                include_top=False)
#         self.model_vgg.predict(np.zeros((1, 224, 224, 3)))

#     def vgg_extract_feat(self, img_path):
#         img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
#         img = image.img_to_array(img)
#         img = np.expand_dims(img, axis=0)
#         img = preprocess_input_vgg(img)
#         feat = self.model_vgg.predict(img)
#         norm_feat = feat[0] / LA.norm(feat[0])
#         norm_feat = [i.item() for i in norm_feat]
#         return norm_feat


# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from numpy import linalg as LA

from torchvision.models import vit_l_32, ViT_L_32_Weights

Image.MAX_IMAGE_PIXELS = None

class ViTNet(object):
    def __init__(self):
        # 定义输入图像的形状
        self.input_shape = (224, 224, 3)
        
        # 加载预训练的 ViT-L/32 模型
        self.model_vit = vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1).to("cuda")
        
        # 移除分类层，保留特征提取部分
        self.model_vit.heads = nn.Identity()
        
        # 将模型设置为评估模式
        self.model_vit.eval()

        # 定义图像预处理的转换
        self.transform = transforms.Compose([
            transforms.Resize((self.input_shape[0], self.input_shape[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def vit_extract_feat(self, img_path):
        # 加载并预处理图像
        """这是从数据库读取网页中的图片"""
        try:
            response = requests.get(img_path)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            """这是从本地读取的图片"""
        except:
            img = Image.open(img_path).convert('RGB')

        img = self.transform(img)
        img = img.unsqueeze(0).to("cuda")  # 增加批次维度
        
        # 提取特征
        with torch.no_grad():
            feat = self.model_vit(img)
        
        feat = feat.to("cpu")
        # 对特征进行 L2 归一化
        norm_feat = feat[0].numpy() / LA.norm(feat[0].numpy())
        norm_feat = [i.item() for i in norm_feat]
        return norm_feat

# 使用示例：
# vit_net = ViTNet()
# features = vit_net.vit_extract_feat("/root/autodl-tmp/pic2piccpp/image/cat.jpg")
# print(features)


# 示例使用
# net = ResNet50Net()
# features = net.resnet_extract_feat('path_to_your_image.jpg')
# print(features)

