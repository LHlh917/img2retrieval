import gradio as gr
import faiss
import h5py
import numpy as np
import sys
sys.path.append('/root/autodl-tmp/image-retrieval-vit')
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
from numpy import linalg as LA

from torchvision.models import vit_l_32, ViT_L_32_Weights

class ViTNet(object):
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.model_vit = vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1).to("cuda")
        self.model_vit.heads = nn.Identity()
        self.model_vit.eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.input_shape[0], self.input_shape[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def vit_extract_feat(self, img):
        img = self.transform(img)
        img = img.unsqueeze(0).to("cuda")
        with torch.no_grad():
            feat = self.model_vit(img)
        feat = feat.to("cpu")
        norm_feat = feat[0].numpy() / LA.norm(feat[0].numpy())
        norm_feat = [i.item() for i in norm_feat]
        return norm_feat

def load_h5_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        feats = np.array(f['dataset_1'])
        names = np.array(f['dataset_2'])
        uk = np.array(f['dataset_3'])
        num_or_code = np.array(f['dataset_4'])
    return feats, names, uk, num_or_code

def search_single_index(h5_path, index_path, query_feats, top_k=5):
    feats, names, uk, num_or_code = load_h5_data(h5_path)
    index = faiss.read_index(index_path)
    D, I = index.search(query_feats, top_k)
    return I, names, D, uk, num_or_code, h5_path

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

def show_similar_images(I, names, D, uk1, num_or_code1, h5_path):
    images = []
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            idx = I[i, j]
            best_match_filename = names[idx].decode('utf-8')
            best_match_id = uk1[idx]
            best_match_serial_num = num_or_code1[idx].decode('utf-8')
            caption = f"ID: {best_match_id}, Serial: {best_match_serial_num}, Source: {os.path.basename(h5_path)}"
            images.append((best_match_filename, caption))
    return images

def retrieve_images(image):
    model = ViTNet()
    query_feats = model.vit_extract_feat(image)
    query_feats = np.array(query_feats).astype('float32')
    if query_feats.ndim == 1:
        query_feats = query_feats.reshape(1, -1)

    h5_files = [
        'index/train-all-sheng.h5',
        'index/train-all-sheng.h5'
    ]
    index_files = [
        'index/faiss_index_1.index',
        'index/faiss_index_1.index'
    ]

    results = distributed_search(h5_files, index_files, query_feats, top_k=3)

    galleries = []
    for I, names, D, uk, num_or_code, h5_path in results:
        images = show_similar_images(I, names, D, uk, num_or_code, h5_path)
        galleries.append([(image, caption) for image, caption in images])  # 返回图片路径和描述
    
    return galleries


css_style = """
body {
    background-color: #f5f5f5;
    font-size: 16px;
}
"""

def main():
    interface = gr.Interface(
        fn=retrieve_images,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Gallery(label="xxxx"), gr.Gallery(label="xxx1")],  # Use multiple galleries for different sources
        title="多省图像检索系统",
        # description="图像检索",
        theme="default",
        css=css_style
    )
    interface.launch(server_name="127.0.0.1", server_port=7861)

if __name__ == "__main__":
    main()
