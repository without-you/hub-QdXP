"""多模态问答接口"""
import os
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from 作业1.app.models.schema import ChatRequest, ChatResponse, SearchResult
from 作业1.app.services.embedding_service import embedding_service
from 作业1.app.services.milvus_service import milvus_service
from 作业1.app.core.config import settings

router = APIRouter(prefix="/api/chat", tags=["问答"])


def get_qwen_client() -> OpenAI:
    """创建Qwen API客户端"""
    return OpenAI(
        api_key=settings.QWEN_API_KEY,
        base_url=settings.QWEN_BASE_URL,
        timeout=120.0
    )


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    多模态问答流程：
    1. 用户提问 → CLIP embedding
    2. Milvus向量检索 top_k 条
    3. 匹配结果 + 用户提问 → Qwen-3.6-plus 生成自然语言答案
    """
    # 1. 将用户问题转为embedding并检索
    query_embedding = embedding_service.embed_text(request.query)
    search_results = milvus_service.search(query_embedding, top_k=request.top_k)

    if not search_results:
        return ChatResponse(
            answer="抱歉，知识库中没有找到与您问题相关的内容。",
            sources=[],
            source_files=[]
        )

    # 2. 转换为SearchResult
    sources = [
        SearchResult(
            chunk_id=r["chunk_id"],
            content=r["content"],
            content_type=r["content_type"],
            source_file=r["source_file"],
            page_number=r["page_number"],
            score=r["score"],
            image_path=r.get("image_path")
        )
        for r in search_results
    ]

    # 3. 构建上下文
    context_parts = []
    for r in sources:
        if r.content_type == "image":
            context_parts.append(f"[图片来源: {os.path.basename(r.source_file)} 第{r.page_number}页]")
        else:
            context_parts.append(f"[{os.path.basename(r.source_file)} 第{r.page_number}页] {r.content}")

    context_text = "\n".join(context_parts)

    # 4. 调用Qwen生成答案
    client = get_qwen_client()
    system_prompt = """你是一个专业的多模态问答助手。请根据提供的上下文信息回答用户的问题。
要求：
1. 准确基于提供的信息进行回答
2. 明确指出答案的信息来源（文件名称和页码）
3. 如果上下文中没有相关信息，诚实地说明无法回答"""

    user_prompt = f"""参考信息：
{context_text}

用户问题: {request.query}"""

    try:
        response = client.chat.completions.create(
            model=settings.QWEN_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=512,
            temperature=0.7
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qwen API调用失败: {str(e)}")

    source_files = list(set(os.path.basename(r.source_file) for r in sources))

    return ChatResponse(
        answer=answer,
        sources=sources,
        source_files=source_files
    )