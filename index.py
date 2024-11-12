# -*- coding: utf-8 -*-

import h5py
import argparse
import numpy as np
from service.VIT import ViTNet
import os
import sys
from os.path import dirname
BASE_DIR = dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import mysql.connector


def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


if __name__ == "__main__":

    """从本地读取图片"""
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--train_data", type=str, default=os.path.join(BASE_DIR, 'data', 'train'), help="train data path.")
    # parser.add_argument("--train_data", type=str, default='/home/image-retrieval-vit/data/train', help="train data path.")
    # # parser.add_argument("--index_file", type=str, default=os.path.join(BASE_DIR, 'index', 'train.h5'), help="index file path.")
    # parser.add_argument("--index_file", type=str, default=os.path.join('/home/image-retrieval-vit', 'index', 'train.h5'), help="index file path.")
    # args = vars(parser.parse_args())
    # img_list = get_imlist(args["train_data"])
    # print("--------------------------------------------------")
    # print("         feature extraction starts")
    # print("--------------------------------------------------")
    # feats = []
    # names = []
    # model = ViTNet()
    # for i, img_path in enumerate(img_list):
    #     norm_feat = model.vit_extract_feat(img_path)
    #     img_name = os.path.split(img_path)[1]
    #     feats.append(norm_feat)
    #     names.append(img_name)
    #     print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))
    # feats = np.array(feats)
    # print("--------------------------------------------------")
    # print("         writing feature extraction results")
    # print("--------------------------------------------------")
    # h5f = h5py.File(args["index_file"], 'w')
    # h5f.create_dataset('dataset_1', data = feats)
    # h5f.create_dataset('dataset_2', data = np.string_(names))
    # h5f.close()

    """从数据库中读取图片"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", type=str, default=os.path.join('/home/image-retrieval-vit', 'index', 'train3.h5'), help="index file path.")
    args = vars(parser.parse_args())
    # 连接到数据库
    cnx = mysql.connector.connect(
        host="192.168.1.104",
        port="3306",
        user="totoro",
        password="123456",
        database="rechecking"
    )

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    feats = []
    names = []
    model = ViTNet()

    # 创建一个游标对象
    cursor = cnx.cursor()

    # 执行一个简单的 SQL 查询
    query = "SELECT file_name,file_path FROM `t_worker_file_info` where file_type = 'work' and file_suffix = 'pic'"
    cursor.execute(query)

    # 获取查询结果
    results = cursor.fetchall()
    i = 0
    # 打印查询结果
    for img_name,row in results[:100]:
        i += 1
        row = "https://jsbqtest-new.oss-cn-hangzhou.aliyuncs.com/" + row
        norm_feat = model.vit_extract_feat(row)
        # img_name = os.path.split(row)[1]

        feats.append(norm_feat)
        names.append(row)
        print("extracting feature from image No. %d , %d images in total" %((i+1), len(results[:100])))

    feats = np.array(feats)
    print("--------------------------------------------------")
    print("         writing feature extraction results")
    print("--------------------------------------------------")
    h5f = h5py.File(args["index_file"], 'w')
    h5f.create_dataset('dataset_1', data = feats)
    # 将字符串列表转换为 NumPy 数组，并使用 h5py.string_dtype 确保 UTF-8 编码
    # names_array = np.array(names, dtype=h5py.string_dtype(encoding='utf-8'))
    h5f.create_dataset('dataset_2', data=np.array(names, dtype=h5py.string_dtype(encoding='utf-8')))
    h5f.close()
