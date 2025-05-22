"""
向量存储管理器
用于创建和查询向量存储
"""
import os
import time
from loguru import logger
from langchain_community.vectorstores import FAISS
from utils.utils import get_all_files
from document_processor import DocumentProcessor
from langchain_huggingface import HuggingFaceEmbeddings

import torch
torch.set_num_threads(8)  # 让PyTorch用8线程（CPU核心数）


class VectorStoreManager:
    """
    向量存储管理器，用于创建和查询向量存储
    """
    def __init__(self, embedding_model, cache_dir, store_path, doc_folder):
        '''
        初始化向量存储管理器
        :param store_path: 向量存储路径
        :param embedding_model: 嵌入模型的id
        :param cache_dir: 模型缓存目录
        注意此处用的 device 是 'cpu'
        '''
        start_time = time.time()
        self.embeddings = HuggingFaceEmbeddings(
                                        model_name=embedding_model,
                                        cache_folder=cache_dir,
                                        model_kwargs={'device': 'cpu'}
                                    )

        self.doc_folder = doc_folder                            
        self.store_path = store_path
        self.vector_store = None
        end_time = time.time()
        logger.info(f"初始化Embedding向量模型耗时: {end_time - start_time:.4f} 秒")
        

    def create_embedding_vector(self, doc_folder):
        """
        如果向量存储库不存在，则根据文档创建它
        20250424: 注意这是处理多个API文档的向量存储
        """
        all_chunks = []
        yaml_files = get_all_files(doc_folder, '.yaml')
        
        for yaml_file in yaml_files:
            md = DocumentProcessor().doc2md(file_path=yaml_file)
            all_chunks.append(md)
        
        self.vector_store = FAISS.from_texts(
            texts=all_chunks,
            embedding=self.embeddings
        )
        self.vector_store.save_local(self.store_path)
        logger.info("向量存储创建成功")

    def load_embedding_vector(self):
        '''
        加载向量存储
        20250424: 注意这是处理多个API文档的向量存储
        '''
        self.vector_store = FAISS.load_local(
            folder_path=self.store_path, 
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("向量存储加载成功")
    
    def main(self, query, k, threshold):
        '''
        主函数，加载向量存储并进行查询
        :param query: 查询文本
        :param k: 返回的结果数量
        :param threshold: 相关性阈值
        '''
        if os.path.exists(self.store_path):
            self.load_embedding_vector()
        else:
            self.create_embedding_vector(self.doc_folder)
            self.load_embedding_vector()


        # 使用相似度搜索
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        similarity_score = [doc[1] for doc in results]
        doc = [doc[0].page_content for doc in results if doc[1] > threshold]

        return doc, similarity_score




if __name__ == "__main__":
    logger.add('vector.log')
    manager = VectorStoreManager(
        embedding_model='BAAI/bge-m3', 
        cache_dir='/data/home/sczc725/run/huggingface/hub',
        store_path='/data/home/sczc725/run/DeepCodeRAG/database/api_name_description_details.faiss',
        doc_folder = '/data/home/sczc725/run/DeepCodeRAG/api_parser/tensorflow/apis_parsed_results', 
    )

    question = '''
    Please assist me in creating a deep learning model for image classification
  using the CIFAR-10 dataset. The model will begin with two convolutional layers followed
  by a max-pooling layer to extract features. The output features will be directly added 
  with the input layer. Finally, these features will be flattened and processed through two
  fully connected layers to produce a probability distribution across the 10 classes.
    '''

    context_chunks, similarity_score = manager.main(question, k=20, threshold=0.1)
    logger.info(f"step3. 查询结果: {context_chunks}")
    logger.info(f"step3. 相关性分数:{similarity_score}")