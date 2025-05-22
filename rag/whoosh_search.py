"""
采用 Whoosh 进行 BM25F 模型的关键词检索
注意，在对单字段进行检索时，BM25F 模型和 BM25 模型的效果是一样的，计算方式也一样
但是在对多字段进行检索时，BM25F 模型的效果会更好
"""

import os
import jieba
import Levenshtein
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
        使用 BM25F 模型进行检索；
        注意：默认为BM25F模型，参数 B=0.75, K1=1.2
        :param query_str: 查询字符串
        :param limit: 返回结果数量限制
        :return: 检索结果列表
        注意：目前设置为 按照 api_name 进行检索，其他信息 只作为匹配后的输出信息。即单字段检索
        """

        # 配置多字段查询解析器
        # field_weights = {
        #     "api_name": 1.0,
        #     "api_description": 1.0,
        #     "api_signature": 1.0,
        #     "api_details": 1.0,
        #     "api_parameters": 1.0,
        #     "api_usage_example": 1.0
        # }
        # parser = MultifieldParser(
        #     list(field_weights.keys()),
        #     schema=self.schema,
        #     fieldboosts=field_weights,
        # )

        # 直接使用 QueryParser 进行单字段检索
        with self.ix.searcher(weighting=BM25F(B=0.75, K1=1.2)) as searcher:
            parser = QueryParser("api_name", schema=self.schema)
            query = parser.parse(query_str)
            results = searcher.search(query, limit=limit, scored=True)

            output = []
            if results:
                print("精确检索成功，结果如下：")
                for hit in results:
                    output.append({
                        "api_name": hit["api_name"],
                        "api_description": hit["api_description"],
                        "api_signature": hit["api_signature"],
                        "api_details": hit["api_details"],
                        "api_usage_description": hit["api_usage_description"],
                        "api_parameters": hit["api_parameters"],
                        "api_usage_example": hit["api_usage_example"],
                        "score": round(hit.score, 4)
                    })
            else:
                print("精确检索失败，正在进行相似度检索...")
                all_docs = searcher.reader().all_stored_fields()  # 返回 generator，每次一个文档
                candidates = []
                for doc in all_docs:
                    name = doc["api_name"]


                    # 方法1:计算 Levenshtein 距离
                    name = name.split('.')[-1]
                    print(f"question: {query_str}, name: {name}")
                    dist = Levenshtein.distance(question, name)
                    similarity = 1 - dist / max(len(question), len(name))

                    # 方法2:计算 Jaccard 相似度
                    # def jaccard_similarity(a, b):
                    #     set_a = set(a)
                    #     set_b = set(b)
                    #     return len(set_a & set_b) / len(set_a | set_b)
                    # similarity = jaccard_similarity(question, name)

                    candidates.append((similarity, doc))

                # 根据相似度进行排序并取 Top K
                jls_extract_var = candidates
                jls_extract_var.sort(reverse=True, key=lambda x: x[0])
                top_k = candidates[:5]
                for _, doc in top_k:
                    output.append({
                        "api_name": doc["api_name"],
                        "api_description": doc["api_description"],
                        "api_signature": doc["api_signature"],
                        "api_details": doc["api_details"],
                        "api_usage_description": doc["api_usage_description"],
                        "api_parameters": doc["api_parameters"],
                        "api_usage_example": doc["api_usage_example"],
                    })

        return output
    
    def main(self, query_str, limit):
        """
        主函数，根据索引是否存在决定是否构建索引，然后进行检索
        """
        if not self.index_created:
            print("索引不存在，正在创建索引...")
            self.add_documents()
            print("索引创建完成，正在进行检索...")
            results = self.search(query_str, limit)
        else:
            print("索引已存在，直接进行检索...")
            results = self.search(query_str, limit)
        
        return results


if __name__ == "__main__":
    searcher = WhooshSearch(
                        index_dir="./database/whoosh_tf_apis_index_0521",
                        docs_path='./api_parser/tensorflow/apis_parsed_results'
                        )

    question = '''
    MaxPooling2D
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
                f"{result['api_description']}\n"
                f"{result['api_signature']}\n\n"
                f"{result['api_details']}\n"
                f"{result['api_usage_description']}\n\n"
                f"{result['api_parameters']}\n\n"
                f"{result['api_usage_example']}\n\n"
            )
        print(api_doc)