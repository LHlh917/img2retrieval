import sys
sys.path.append('/root/autodl-tmp/image-retrieval-vit')

import faiss
import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from service.VIT import ViTNet


# 加载H5文件中的数据
def load_h5_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        feats = np.array(f['dataset_1'])
        names = np.array(f['dataset_2'])
        uk = np.array(f['dataset_3'])
        num_or_code = np.array(f['dataset_4'])
    return feats, names, uk, num_or_code

# 搜索单个H5文件的索引并返回文件路径
def search_single_index(h5_path, index_path, query_feats, top_k=5):
    feats, names, uk, num_or_code = load_h5_data(h5_path)
    index = faiss.read_index(index_path)
    D, I = index.search(query_feats, top_k)
    return I, names, D, uk, num_or_code, h5_path  # 返回时包括 h5 文件的路径

# 分布式处理函数，返回每个H5文件的结果及其来源
def distributed_search(h5_paths, index_paths, query_feats, top_k=5):
    results = []
    with ThreadPoolExecutor(max_workers=len(h5_paths)) as executor:
        futures = [
            executor.submit(search_single_index, h5_paths[i], index_paths[i], query_feats, top_k)
            for i in range(len(h5_paths))
        ]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error occurred: {e}")
    
    return results

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



# 用户输入图片路径并提取特征
model = ViTNet()
path = '/root/autodl-tmp/image.jpg'
query_feats = model.vit_extract_feat(path)   # 提取图片特征
query_feats = np.array(query_feats).astype('float32')
if query_feats.ndim == 1:
    query_feats = query_feats.reshape(1, -1)

# H5文件和索引路径列表
h5_files = [
    '/root/autodl-tmp/image-retrieval-vit/index/train-970000-sheng.h5',
    '/root/autodl-tmp/image-retrieval-vit/index/train-all-sheng.h5'
]
index_files = [
    '/root/autodl-tmp/text-image-show/image_retrieval_vit/index/daopai_index.index',
    '/root/autodl-tmp/image-retrieval-vit/index/faiss_index_1.index'
]

# 分布式检索
results = distributed_search(h5_files, index_files, query_feats, top_k=3)

# 显示每个H5文件的检索结果及其来源
for I, names, D, uk, num_or_code, h5_path in results:
    print(f"Results from {h5_path}:")
    output_images, uk_result, num_or_code_result = show_similar_images(I, names, D, uk, num_or_code)
    for i in range(len(output_images)):
        print(f"Image {i+1}: {output_images[i]} | ID: {uk_result[i]} | Serial Number: {num_or_code_result[i]}")
    print("------------------------------------------")
