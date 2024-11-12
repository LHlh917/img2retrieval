import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from service.VIT import ViTNet
from service.numpy_retrieval import NumpyRetrieval
from service.faiss_retrieval import FaissRetrieval
from service.es_retrieval import ESRetrieval
from service.milvus_retrieval import MilvusRetrieval
import os
import requests
from PIL import Image
from io import BytesIO
import time

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 定义图像检索引擎
class RetrievalEngine(object):

    def __init__(self, index_file, db_name):
        self.index_file = index_file
        self.db_name = db_name
        self.numpy_r = self.faiss_r = self.es_r = self.milvus_r = None

    def get_method(self, m_name):
        m_name = "%s_handler" % str(m_name)
        method = getattr(self, m_name, self.default_handler)
        return method

    def numpy_handler(self, query_vector, req_id=None):
        if self.numpy_r is None:
            self.numpy_r = NumpyRetrieval(self.index_file)
        return self.numpy_r.retrieve(query_vector)

    def faiss_handler(self, query_vector, req_id=None):
        if self.faiss_r is None:
            self.faiss_r = FaissRetrieval(self.index_file)
        return self.faiss_r.retrieve(query_vector)

    def es_handler(self, query_vector, req_id=None):
        if self.es_r is None:
            self.es_r = ESRetrieval(self.db_name, self.index_file)
        return self.es_r.retrieve(query_vector)

    def milvus_handler(self, query_vector, req_id=None):
        if self.milvus_r is None:
            self.milvus_r = MilvusRetrieval(self.db_name, self.index_file)
        return self.milvus_r.retrieve(query_vector)

    def default_handler(self, query_vector, req_id=None):
        return []

# 加载ResNet50模型
model = ViTNet()

# 定义处理函数
def image_retrieval(query_image, db_name):
    engine = 'faiss'
    # index_file = os.path.join(os.getcwd(), 'index', 'train-all-sheng.h5')
    index_file = os.path.join(os.getcwd(), 'index', '/root/autodl-tmp/image-retrieval-vit/index/train-970000-sheng.h5')

    # 提取查询图像的特征
    t = time.time()
    query_vector = model.vit_extract_feat(query_image)
    
    # 初始化检索引擎
    re = RetrievalEngine(index_file, db_name)
    
    # 根据选定的检索引擎执行检索
    results = re.get_method(engine)(query_vector, None)
    
    # 处理检索结果，获取前三个结果
    top_results = results[:3]
    print('检索结果:',top_results)
    output_images = []
    for result in top_results:
        image_name = result['name'].decode('utf-8')
        """从本地加载图片方式"""
        # image_path = os.path.join('/home/image-retrieval/data/train', image_name)  # 替换为你实际图片存放的路径
        # output_images.append(Image.open(image_path))
        """从数据库中加载图片方式"""
        if image_name.startswith('http://') or image_name.startswith('https://'):
        # 如果 image_name 是一个网址，下载图片
            response = requests.get(image_name)
            image = Image.open(BytesIO(response.content))
        else:
        # 如果 image_name 是本地路径，直接打开图片
            image = Image.open(image_name)    

        output_images.append(image)
        # output_images.append(Image.open(image_name))
    print("**************************")
    print(time.time() - t)
    return output_images

# 设置Gradio界面
db_name = 'image_retrieval'
interface = gr.Interface(
    fn=image_retrieval,
    inputs=[
        gr.Image(type="filepath", label="查询图像"),
        gr.Textbox(value=db_name, label="Database Name"),
        # gr.Radio(choices=['numpy', 'faiss', 'es', 'milvus'], label="Retrieval Engine", value='faiss')
    ],
    outputs=[gr.Image(label="检索结果 1"), gr.Image(label="检索结果 2"), gr.Image(label="检索结果 3")],
    title="图像检索系统",
    description="上传查询图像并选择检索引擎，获取相似图像列表。"
)

# 启动Gradio界面
if __name__ == "__main__":
    
    interface.launch()

    
