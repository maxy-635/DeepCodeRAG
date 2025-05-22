"""
采用 Whoosh 进行 BM25F 模型的关键词检索
注意，在对单字段进行检索时，BM25F 模型和 BM25 模型的效果是一样的，计算方式也一样
但是在对多字段进行检索时，BM25F 模型的效果会更好
"""

import os
import jieba
import Levenshtein
from rouge import rouge_l_sentence_level
from loguru import logger
from document_processor import DocumentProcessor
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
from whoosh.qparser import MultifieldParser
from whoosh import scoring
from whoosh.analysis import Token, Analyzer
from whoosh.scoring import BM25F
from utils.utils import get_all_files


class JiebaAnalyzer(Analyzer):
    """
    使用 Jieba 分词器的自定义 Analyzer
    """
    def __call__(self, value, positions=True, chars=True,
                 keeporiginal=False, removestops=True, start_pos=0,
                 start_char=0, mode='', **kwargs):
        t = Token(positions, chars, removestops=removestops,
                  mode=mode, **kwargs)

        pos = start_pos
        char_pos = start_char
        for word in jieba.cut(value):
            word = word.strip()
            if not word:
                continue
            t.original = t.text = word
            if positions:
                t.pos = pos
                pos += 1
            if chars:
                t.startchar = char_pos
                t.endchar = char_pos + len(word)
                char_pos = t.endchar
            yield t

class WhooshSearch(JiebaAnalyzer):
    """
    Whoosh 搜索类，使用 BM25 模型进行 关键词 检索
    """
    def __init__(self, docs_path, index_dir):
        """
        初始化 Whoosh 搜索类
        :param docs_path: API文档路径
        :param index_dir: 要创建的数据库索引目录路径
        """
        self.docs_path = docs_path
        self.index_dir = index_dir
        
        self.schema = Schema(
            api_name = TEXT(stored=True, analyzer=JiebaAnalyzer()),
            api_description = TEXT(stored=True, analyzer=JiebaAnalyzer()),
            api_signature = TEXT(stored=True, analyzer=JiebaAnalyzer()),
            api_details = TEXT(stored=True, analyzer=JiebaAnalyzer()),
            api_usage_description = TEXT(stored=True, analyzer=JiebaAnalyzer()),
            api_parameters = TEXT(stored=True, analyzer=JiebaAnalyzer()),
            api_usage_example = TEXT(stored=True, analyzer=JiebaAnalyzer()),
        )

        # 初始化索引目录，如果不存在则创建
        if not os.path.exists(self.index_dir):
            os.mkdir(self.index_dir)
            self.ix = create_in(self.index_dir, self.schema)
            self.index_created = False
        # 如果索引目录已经存在，则打开它
        else:
            self.ix = open_dir(self.index_dir)
            self.index_created = True

    def add_documents(self):
        """
        添加文档到索引
        :param doc_folder: 文档目录
        """
        writer = self.ix.writer()

        for yaml_file in get_all_files(self.docs_path, '.yaml'):
            api_name, api_description, api_signatures, api_details, api_usage_description, api_parameters, api_usage = DocumentProcessor().doc2str(yaml_file)
            writer.add_document(
                api_name=api_name,
                api_description=api_description,
                api_signature=api_signatures,
                api_details=api_details,
                api_usage_description=api_usage_description,
                api_parameters=api_parameters,
                api_usage_example=api_usage
            )

        writer.commit()


    def search(self, query_str, limit):
        """
        使用 BM25F 模型进行检索，默认为单字段（api_name）检索。
        如果没有检索到结果，则使用 ROUGE-L 相似度作为回退方案。
        
        :param query_str: 用户输入的查询字符串
        :param limit: 最多返回多少条结果
        :return: 检索结果组成的列表，每条结果包含字段信息和分数
        """

        def format_result(data, score):
            """构造统一格式的输出结果。"""
            return {
                "api_name": data["api_name"],
                "api_description": data["api_description"],
                "api_signature": data["api_signature"],
                "api_details": data["api_details"],
                "api_usage_description": data["api_usage_description"],
                "api_parameters": data["api_parameters"],
                "api_usage_example": data["api_usage_example"],
                "score": round(score, 4)
            }

        with self.ix.searcher(weighting=BM25F(B=0.75, K1=1.2)) as searcher:
            # 使用 QueryParser 对查询字符串进行解析
            parser = QueryParser("api_name", schema=self.schema)
            query = parser.parse(query_str)

            # 执行 BM25F 精确搜索
            results = searcher.search(query, limit=limit, scored=True)
            if results:
                logger.info(f"{query}精确检索成功")
                # 将搜索结果格式化并返回
                output = [format_result(result, result.score) for result in results]

            else:
                logger.info(f"{query}精确检索失败，进行相似度检索")
                # 读取所有已存储的文档，进行相似度比较
                all_docs = searcher.reader().all_stored_fields()
                candidates = []
                for doc in all_docs:
                    api_name = doc["api_name"]
                    # 使用 ROUGE-L 计算相似度（只用 f1-score）
                    _, _, f1 = rouge_l_sentence_level(summary_sentence=query_str, reference_sentence=api_name)
                    candidates.append((f1, doc))

                # 对候选文档按相似度从高到低排序，选出前 limit 个
                candidates.sort(key=lambda item: item[0], reverse=True)
                top_docs = candidates[:limit]

                logger.info(f"{query}相似度检索成功")

                output = [format_result(doc, score) for score, doc in top_docs]

            return output

    
    def main(self, query_str, limit):
        """
        主函数，根据索引是否存在决定是否构建索引，然后进行检索
        """
        if not self.index_created:
            logger.info("索引不存在，正在创建索引...")
            self.add_documents()
            logger.info("索引创建完成，正在进行检索...")
            results = self.search(query_str, limit)
        else:
            logger.info("索引已存在，直接进行检索...")
            results = self.search(query_str, limit)
        
        return results


if __name__ == "__main__":
    searcher = WhooshSearch(
                        index_dir="./database/whoosh_tf_apis_index_0521",
                        docs_path='./api_parser/tensorflow/apis_parsed_results'
                        )

    question = '''
tensorflow.keras.layers.AveragePool2D
'''
    results = searcher.main(
        query_str=question,
        limit=3
    )
    # 输出检索结果
    print(f"检索到 {len(results)} 条结果：")
    for result in results:
        api_doc = (
                f"{result['api_name']}\n\n"
                # f"{result['api_description']}\n"
                # f"{result['api_signature']}\n\n"
                # f"{result['api_details']}\n"
                # f"{result['api_usage_description']}\n\n"
                # f"{result['api_parameters']}\n\n"
                # f"{result['api_usage_example']}\n\n"
                f"相似度分数: {result['score']}\n\n"
            )
        print(api_doc)