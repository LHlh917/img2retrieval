import faiss
import h5py
import numpy as np
import requests
from PIL import Image
from io import BytesIO


def show_similar_images(I, names, D, uk1, num_or_code1):
    image_path = []
    id = []
    serial_num = []
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            idx = I[i, j]
            best_match_filename = names[idx].decode('utf-8')
            best_match_id = uk1[idx]
            best_match_serial_num = num_or_code1[idx].decode('utf-8')
            image_path.append(best_match_filename)
            id.append(best_match_id)
            serial_num.append(best_match_serial_num)
    return image_path, id, serial_num

### 用户输入模块
# path = input("请输入文件路径：")
path = '/root/autodl-tmp/image.jpg'

### 模型定义模块
from service.VIT import ViTNet
model = ViTNet()

### 处理用户输入图片模块
query_feats = model.vit_extract_feat(path)   # 这里输出的是图片的特征，维度是[1,1024]

index = 'IVF'

### 现在是加载各个省份的h5文件和索引结构
if  index == 'IVF':
    hunan_index = '/root/autodl-tmp/text-image-show/image_retrieval_vit/index/daopai_index.index'
if index == 'lsh':
    hunan_index = '/root/autodl-tmp/text-image-show/image_retrieval_vit/index/lsh_index.index'
index = faiss.read_index(hunan_index)
hunan_h5 = '/root/autodl-tmp/image-retrieval-vit/index/train-970000-sheng.h5'
with h5py.File(hunan_h5, 'r') as f:
    feats = np.array(f['dataset_1'])
    names = np.array(f['dataset_2'])
    uk = np.array(f['dataset_3'])               # uk是图片的id
    num_or_code = np.array(f['dataset_4'])      # num_or_code是图片的流水号

### 查询模块
query_feats = np.array(query_feats).astype('float32')
if query_feats.ndim == 1:
    query_feats = query_feats.reshape(1, -1)
D, I = index.search(query_feats, 5)  

output_images,uk,num_or_code = show_similar_images(I, names, D, uk, num_or_code)



print(
        output_images[0], uk[0],num_or_code[0],
        "-------------------------------------",
        output_images[1], uk[1],num_or_code[1],
        "-------------------------------------",
        output_images[2], uk[2],num_or_code[2],
        # f"Retrieval Time: {retrieval_time:.2f} seconds"
    )
