# 多模态RAG系统

基于检索增强生成（RAG）的多模态问答系统，支持PDF文档解析、跨模态检索和图文推理问答。

## 技术架构

```
用户上传PDF
    ↓
FastAPI (上传接口)
    ↓
Kafka (topic: document_parse)
    ↓
Worker (消费kafka)
    ↓
Mineru (解析PDF) → SQLite (元数据) + Milvus (向量)
    ↓
用户提问 → CLIP embedding → Milvus检索 → Qwen-3.6-plus 生成答案
```

## 核心组件

| 组件 | 用途 |
|------|------|
| Kafka | 异步解耦，文档上传后发送到队列 |
| Mineru | 远程API解析PDF为markdown+图片 |
| SQLite | 存储文档元数据、chunk信息 |
| Milvus | 存储向量，CLIP embedding检索 |
| CLIP | 文本/图像嵌入 |
| Qwen-3.6-plus | 云端大模型生成自然语言答案 |

## 启动方式

```bash
# 1. 启动API服务
python -m app.main

# 2. 启动Worker（另一个终端）
python -m app.workers.document_worker
```

## API接口

### 文档上传
```bash
POST /api/document/upload
# 上传PDF → 写入Kafka → Worker异步处理
```

### 文档列表

```bash
GET /api/document/list
GET /api/document/{document_id}
```

### 问答
```bash
POST /api/chat
{
  "query": "问题内容",
  "top_k": 5
}
# 返回: 答案 + 匹配来源 + 来源文件列表
```

## 环境变量

```bash
# Milvus (Zilliz Cloud)
MILVUS_URI=https://in03-5cb3b56f3af9ebc.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn
MILVUS_TOKEN=9027d285f...

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_DOCUMENT=document_parse

# Mineru
MINERU_API_KEY=your_key

# Qwen
QWEN_API_KEY=sk-777ae59d8b3e451db4dd91fe6961dbe5
QWEN_MODEL_NAME=qwen-3.6-35b-a3b
```