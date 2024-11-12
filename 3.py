import mysql.connector
import os
import torch
import torch.nn as nn
import argparse
import h5py
import numpy as np
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import requests
from torchvision import models, transforms
from torchvision.models import vit_l_32, ViT_L_32_Weights
import time

Image.MAX_IMAGE_PIXELS = None

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

    def vit_extract_feat_batch(self, img_paths):
        imgs = []
        failed_images = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            img_data = list(executor.map(self.load_image, img_paths))
        
        for img, path in zip(img_data, img_paths):
            if img is not None:
                imgs.append(self.transform(img))
            else:
                failed_images.append(path)

        if len(imgs) == 0:
            return [], failed_images

        imgs = torch.stack(imgs).to("cuda:0")
        with torch.no_grad():
            feats = self.model_vit(imgs)
        feats = feats.to("cpu")

        norm_feats = []
        for feat in feats:
            norm_feat = feat.numpy() / np.linalg.norm(feat.numpy())
            norm_feats.append(norm_feat)
        return norm_feats, failed_images

    def load_image(self, img_path):
        try:
            response = requests.get(img_path)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return img
        except:
            print(f"Failed to load image: {img_path}")
            return None

def save_to_h5(feats, names, index_file):
    with h5py.File(index_file, 'w') as h5f:
        h5f.create_dataset('dataset_1', data=feats)
        h5f.create_dataset('dataset_2', data=np.array(names, dtype=h5py.string_dtype(encoding='utf-8')))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", type=str, default=os.path.join('/home/image-retrieval-vit', 'index', 'train4.h5'), help="index file path.")
    args = vars(parser.parse_args())

    cnx = mysql.connector.connect(
        host="127.0.0.1",
        port="3306",
        user="root",
        password="yikData@0829",
        database="rechecking"
    )

    feats = []
    names = []
    failed_count = 0
    processed_count = 0
    model = ViTNet()

    cursor = cnx.cursor()
    query = "SELECT file_name, file_path FROM `t_worker_file_info` WHERE file_type = 'work' AND file_suffix = 'pic'"
    cursor.execute(query)

    results = cursor.fetchall()

    batch_size = 256
    save_interval = 10000  # 每处理10000个数据保存一次

    for i in range(0, len(results), batch_size):
        batch_results = results[i:i+batch_size]
        img_paths = ["https://jsbqtest-new.oss-cn-hangzhou.aliyuncs.com/" + row[1] for row in batch_results]
        batch_feats, failed_images = model.vit_extract_feat_batch(img_paths)
        failed_count += len(failed_images)
        feats.extend(batch_feats)
        successful_names = ["https://jsbqtest-new.oss-cn-hangzhou.aliyuncs.com/" + row[1] for row, img_path in zip(batch_results, img_paths) if img_path not in failed_images]
        print(successful_names)
        names.extend(successful_names)

        processed_count += len(batch_results)
        print(f"Extracting features from image No. {processed_count} of {len(results)}")

        # 保存逻辑
        # if processed_count >= save_interval and (processed_count // save_interval) * save_interval - save_interval < len(feats):
        if processed_count >= save_interval:
            print(f"Saving progress at {processed_count} images.")
            save_to_h5(np.array(feats), names, args["index_file"])
            save_interval += 10000
            print(f"save_interval at {save_interval}")

    # 最后一次保存处理的所有数据
    if len(feats) > 0:
        print(f"Saving final progress at {processed_count} images.")
        save_to_h5(np.array(feats), names, args["index_file"])

    print(f"Total failed images: {failed_count}")

if __name__ == "__main__":
    t = time.time()
    main()
    print(time.time() - t)
