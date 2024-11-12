import faiss
import h5py
import numpy as np
import argparse
from service.VIT import ViTNet

import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO


def load_features(index_file):
    """
    从 HDF5 文件中加载图像特征和名称
    """
    with h5py.File(index_file, 'r') as f:
        feats = np.array(f['dataset_1'])
        names = np.array(f['dataset_2'])
        id = f['dataset_3']
        retrieval_db_serial = f["dataset_4"]
        serial = [serial.decode('utf-8') for serial in retrieval_db_serial]
    return feats, names,id,serial

def build_faiss_index(feats, nlist=100):
    """
    使用 FAISS 创建并训练索引
    """
    d = feats.shape[1]
    
    # 确保特征矩阵的数据类型为 float32
    feats = feats.astype('float32')

    # 创建一个 L2 距离的平面索引
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    
    # 训练索引
    index.train(feats)
    
    # 添加特征到索引
    index.add(feats)
    
    return index

def save_faiss_index(index, file_path):
    """
    将 FAISS 索引保存到文件
    """
    faiss.write_index(index, file_path)

def load_faiss_index(file_path):
    """
    从文件中加载 FAISS 索引
    """
    return faiss.read_index(file_path)

def search(index, query_feats, k=5):
    """
    在 FAISS 索引中进行检索
    """
    query_feats = np.array(query_feats).astype('float32')
    
    # 如果 query_feats 是一维数组，将其转换为二维数组
    if query_feats.ndim == 1:
        query_feats = query_feats[None, :]  # 将一维数组转换为二维数组

    D, I = index.search(query_feats, k)
    return D, I

def print_similar_images(indices, names):
    """
    打印与查询图像相似的图像名称或路径
    """
    for i, index_list in enumerate(indices):
        print(f"Query Image {i + 1} similar images:")
        for idx in index_list:
            print(f"- {names[idx]}")
        print("\n")

def fetch_image(image_url):
    """
    从 URL 获取图像
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def show_similar_images(indices, names):
    """
    显示与查询图像相似的图像
    """
    num_queries = len(indices)
    num_neighbors = len(indices[0])

    # 处理 axes 为一维或二维的情况
    if num_queries == 1:
        fig, axes = plt.subplots(1, num_neighbors, figsize=(15, 15))
        axes = np.expand_dims(axes, axis=0)  # 变成二维数组方便索引
    else:
        fig, axes = plt.subplots(num_queries, num_neighbors, figsize=(15, 15))
    
    for i, index_list in enumerate(indices):
        for j, idx in enumerate(index_list):
            img = fetch_image(names[idx])
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(f"Query {i + 1}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", type=str, default='index/train-all-sheng.h5', help="HDF5 index file path.")
    parser.add_argument("--faiss_index_file", type=str, default='index/faiss_index_1.index', help="FAISS index file path.")
    args = vars(parser.parse_args())

    # 加载特征和名称
    feats, names,id,serial = load_features(args["index_file"])

    # 构建 FAISS 索引
    nlist = 2  # 可以根据数据量调整
    index = build_faiss_index(feats, nlist)

    # 保存 FAISS 索引
    save_faiss_index(index, args["faiss_index_file"])

    # 进行检索
    query_image_paths = 'data/train/001_accordion_image_0003.jpg'  # 替换为实际图像路径
    model = ViTNet()  # 使用之前定义的 ViTNet 类
    query_feats = model.vit_extract_feat(query_image_paths)
    faiss_index = load_faiss_index(args["faiss_index_file"])
    distances, indices = search(faiss_index, np.array(query_feats))
    
    print("Distances:\n", distances)
    print("Indices:\n", indices)

    # 打印相似图像信息
    print_similar_images(indices, names)

    # 可视化相似图像
    show_similar_images(indices, names)


if __name__ == "__main__":
    main()
