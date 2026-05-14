import pdfplumber  # 用于从 PDF 中提取文本
import re  # 正则表达式，用于智能分句
from typing import List, Dict
import os  # 用于检查文件路径是否存在

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY = "sk-777ae59d8b3e451db4dd91fe6961dbe5"
LLM_MODEL_NAME = "deepseek-v4-flash"
MODEL_PATH = "E:/models/BAAI/bge-small-zh-v1.5"
PDF_PATH = "./docs/清华大学2026HermesAgent深度研究报告60页.pdf"
# 1、对文档进行分块
def split_sentence(docs, max_chunk=300):
    """文档解析"""
    # 如果输入为空或全是空白字符，直接返回空列表
    if not docs or not docs.strip():
        return []

    # 按换行符分割成段落，并过滤掉空段落，同时去除首尾空白
    paragraphs = [p.strip() for p in docs.strip().split('\n') if p.strip()]
    chunks = [] #用于存储最终分割完成的文本块

    #遍历每一个段落
    for paragraph in paragraphs:
        # 使用正则表达式按中文句号、感叹号、问号、分号进行切分
        sentences = re.split(r'(?<=[.?!])\s', paragraph)
        current_chunk = ""   # 当前正在构建的文本块

        # 遍历该段落中的每一个句子
        for sentence in sentences:
            sentence = sentence.strip()  # 去除句子首尾空白
            if not sentence:   # 跳过空句子
                continue

            # 判断：如果把当前句子加入 current_chunk 后不超过最大长度
            if len(current_chunk) + len(sentence) <= max_chunk:
                current_chunk += sentence
            else:
                # 如果加上会超长，则先保存当前块（如果非空）
                if current_chunk:
                    chunks.append(current_chunk)
                # 开启新块，从当前句子开始（即使它自己就很长）
                current_chunk = sentence

        # 处理段落末尾剩余的内容（最后一块）
        if current_chunk:
            chunks.append(current_chunk)
    return chunks

# 2. 加载 PDF 文件并将其内容分块

def load_chunk_pdf(pdf_path:str) -> List[Dict[str, str]]:
    """PDF加载"""
    if not os.path.isfile(pdf_path):
       raise FileNotFoundError("PDF file not found.")

    print("加载pdf中")

    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"  # 用双换行分隔页面

    # 使用智能分块器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每块约500字符
        chunk_overlap=100,  # 相邻块重叠100字符，防止割裂
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_text(full_text)

    # 过滤掉太短或无意义的块（如纯数字、页码）
    filtered_chunks = [
        chunk.strip()
        for chunk in chunks
        if len(chunk.strip()) > 20  # 至少20个字符
           and not chunk.strip().isdigit()  # 排除纯数字（如页码）
    ]

    return [{"content": chunk} for chunk in filtered_chunks]

#3.创建FAISS向量数据库 更加兼容langchain方便使用
def create_vectorstore(pdf_content:List[Dict], model_path:str):
    """
       将已有的文本块和向量，转换为 LangChain 的 FAISS 向量数据库。
    """
    # 1.将数据结构转化为langchain的Document结构
    documents = [
        Document(
            page_content=item["content"],
        )
        for item in pdf_content
    ]
    #2.创建embedding容器
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device":'cuda'},
    )
    # 3.创建索引
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore

# 4.rag调用链条
def create_rag_chain(vectorstore:FAISS):
    """使用已创建的 vectorstore 构建 RAG 链。"""
    # 1.创建检索器
    retriever = vectorstore.as_retriever(sercher_kwargs={'k': 5})
    # 2.初始化llm
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME  ,  # 模型的代号
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )
    # 3.提示词模板
    prompt_template ="""
    你是一个专家，你擅长回答用户提问，帮我结合给定的资料，回答下面的问题。
    如果问题无法从资料中获得，请回答无法回答。
    资料：
    {context}
    问题：{question}"""
    # 4.提问llm
    def ask_question(user_question:str):
        # 1.检索文档
        document = retriever.invoke(user_question)
        context = "\n\n".join([doc.page_content for doc in document])
        # 2.填充模板
        full_prompt = prompt_template.format(context=context, question=user_question)
        # 3.调用大语言模型
        ai_message = llm.invoke([HumanMessage(content=full_prompt)])
        # 4.获取返回结果
        answer = ai_message.content
        return answer

    return ask_question

if __name__ == '__main__':
    # 文档相对路径
    pdf_path = PDF_PATH
    # 解析文档
    pdf_content = load_chunk_pdf(pdf_path)
    print("总块数:", len(pdf_content))
    print("\n前5块内容预览:")
    for i in range(min(5, len(pdf_content))):
        print(f"--- 块 {i} ---")
        print(pdf_content[i]['content'][:200])  # 打印前200字符
    # 本地模型位置
    model_path = MODEL_PATH
    model = SentenceTransformer(model_path)
    texts = [item['content'] for item in pdf_content]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    # 向量化操作
    vectorstore = create_vectorstore(pdf_content, model_path)
    # 生成rag调用链条
    ask_question = create_rag_chain(vectorstore)

    while True:
        user_question = input("\n请输入您的问题 (输入 'quit' 退出): ")
        if user_question.lower() == 'quit':
            break
        print("检索思考中...")
        answer = ask_question(user_question)
        print(f"\n答案: {answer}")