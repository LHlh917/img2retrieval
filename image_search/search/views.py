from django.shortcuts import render
from .forms import ImageSearchForm
from .models import Image
import faiss
import numpy as np
from PIL import Image as PilImage
import torch
import torchvision.transforms as transforms

import sys
sys.path.append("/home/image-retrieval")

import os
from django.conf import settings
from service.VIT import ResNet50Net
from service.faiss_retrieval import FaissRetrieval

# 假设你有一个PyTorch模型用于提取特征
# model = torch.load('path_to_your_model.pth')
model = ResNet50Net()
# model.eval()

# 加载FAISS索引
# index = faiss.read_index('/home/image-retrieval/index/train.h5')
faiss_retrieval = FaissRetrieval('/home/image-retrieval/index/train.h5')


def extract_features(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 假设模型输入尺寸是224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # 添加batch维度
    # with torch.no_grad():
    #     features = model.resnet_extract_feat(image)
    features = model.resnet_extract_feat(image)
    features = np.array(features)
    # return features.numpy().flatten()
    return features.flatten()

# def index(request):
#     if request.method == 'POST':
#         form = ImageSearchForm(request.POST, request.FILES)
#         if form.is_valid():
#             uploaded_image = PilImage.open(form.cleaned_data['image'])
#             # 打印类型和其他信息
#             print(type(uploaded_image))
#             # 如果是文件对象，直接读取
#             if hasattr(uploaded_image, 'read'):
#                 from io import BytesIO
#                 image = PilImage.open(BytesIO(uploaded_image.read()))
#             else:
#                 # 如果已经是 PIL Image 对象，直接使用
#                 image = uploaded_image


#             features = extract_features(image)
#             # features = np.expand_dims(features, axis=0).astype('float32')

#             # # 使用FAISS进行检索
#             # D, I = index.search(features, 3)  # 检索最相似的3个图片
#             # similar_images = Image.objects.filter(id__in=I.flatten())
#             # 使用FaissRetrieval检索最相似的图片
#             results = faiss_retrieval.retrieve(features, search_size=3)
#             similar_images = [Image.objects.get(title=res['name']) for res in results]


#             return render(request, 'search/results.html', {'images': similar_images, 'distances': [res['score'] for res in results]})

#     else:
#         form = ImageSearchForm()

#     return render(request, 'search/index.html', {'form': form})


def index(request):
    if request.method == 'POST':
        form = ImageSearchForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = PilImage.open(form.cleaned_data['image'])
            features = extract_features(uploaded_image)
            results = faiss_retrieval.retrieve(features, search_size=3)

            # Prepare to display images
            image_list = []
            for res in results:
                image_path = os.path.join(settings.MEDIA_ROOT, res['name'].decode('utf-8'))
                
                if os.path.exists(image_path):
                    img_url = os.path.join(settings.MEDIA_URL, res['name'].decode('utf-8'))
                    image_list.append((img_url, res['score']))
                else:
                    print(f"Image {res['name'].decode('utf-8')} does not exist")

            combined_list = image_list
            print(f"Combined list: {combined_list}")  # For debugging

            return render(request, 'search/results.html', {'combined_list': combined_list})
    else:
        form = ImageSearchForm()

    return render(request, 'search/index.html', {'form': form})







