from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncIterable, List, Optional, Literal

from fastapi import Body, Request
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.prompts.chat import ChatPromptTemplate

from chatchat.settings import Settings
from chatchat.server.agent.tools_factory.search_internet import search_engine
from chatchat.server.api_server.api_schemas import OpenAIChatOutput
from chatchat.server.chat.utils import History
from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory
from chatchat.server.knowledge_base.kb_doc_api import search_docs, search_temp_docs
from chatchat.server.knowledge_base.utils import format_reference
from chatchat.server.utils import (
    wrap_done,
    get_ChatOpenAI,
    get_default_llm,
    BaseResponse,
    get_prompt_template,
    build_logger,
    check_embed_model,
    api_address,
)

logger = build_logger()

async def kb_chat(
    query: str = Body(..., description="用户输入", examples=["你好"]),
    mode: Literal["local_kb", "temp_kb", "search_engine"] = Body("local_kb", description="知识来源"),
    kb_name: str = Body("", description="mode=local_kb时为知识库名称；temp_kb时为临时知识库ID，search_engine时为搜索引擎名称", examples=["samples"]),
    top_k: int = Body(Settings.kb_settings.VECTOR_SEARCH_TOP_K, description="匹配向量数"),
    score_threshold: float = Body(
        Settings.kb_settings.SCORE_THRESHOLD,
        description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
        ge=0,
        le=2,
    ),
    history: List[History] = Body(
        [],
        description="历史对话",
        examples=[
            [
                {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                {"role": "assistant", "content": "虎头虎脑"},
            ]
        ],
    ),
    stream: bool = Body(True, description="流式输出"),
    model: str = Body(get_default_llm(), description="LLM 模型名称。"),
    temperature: float = Body(Settings.model_settings.TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
    max_tokens: Optional[int] = Body(
        Settings.model_settings.MAX_TOKENS,
        description="限制LLM生成Token数量，默认None代表模型最大值",
    ),
    prompt_name: str = Body("default", description="使用的prompt模板名称(在prompt_settings.yaml中配置)"),
    return_direct: bool = Body(False, description="直接返回检索结果，不送入 LLM"),
    request: Request = None,
):
    # 确保 max_tokens 和 prompt_name 都有默认值
    max_tokens = max_tokens or Settings.model_settings.MAX_TOKENS
    prompt_name = prompt_name or "default"

    # 确保历史记录是每个请求独立的
    history = [History.from_data(h) for h in history]

    # 获取对应的知识库或搜索引擎
    if mode == "local_kb":
        kb = KBServiceFactory.get_service_by_name(kb_name)
        if kb is None:
            return BaseResponse(code=404, msg=f"未找到知识库 {kb_name}")
    elif mode == "temp_kb":
        kb = NoneAsyncIteratorCallbackHandler
        kb = None

    # 异步处理多个任务并发请求
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        prompt_name = "default"
        try:
            if mode == "local_kb":
                ok, msg = kb.check_embed_model()
                if not ok:
                    raise ValueError(msg)
                docs = await run_in_threadpool(
                    search_docs,
                    query=query,
                    knowledge_base_name=kb_name,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    file_name="",
                    metadata={},
                )
                source_documents = format_reference(kb_name, docs, api_address(is_public=True))
            elif mode == "temp_kb":
                ok, msg = check_embed_model()
                if not ok:
                    raise ValueError(msg)
                docs = await run_in_threadpool(
                    search_temp_docs, kb_name, query=query, top_k=top_k, score_threshold=score_threshold
                )
                source_documents = format_reference(kb_name, docs, api_address(is_public=True))
            elif mode == "search_engine":
                result = await run_in_threadpool(search_engine, query, top_k, kb_name)
                docs = [x.dict() for x in result.get("docs", [])]
                source_documents = [
                    f"""出处 [{i + 1}] [{d['metadata']['filename']}]({d['metadata']['source']}) \n\n{d['page_content']}\n\n"""
                    for i, d in enumerate(docs)
                ]
            else:
                docs = []
                source_documents = []

            # 设置默认 prompt_name
            if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
                prompt_name = "empty"

            if return_direct:
                yield OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    model=None,
                    object="chat.completion",
                    content="",
                    role="assistant",
                    finish_reason="stop",
                    docs=source_documents,
                ).model_dump_json()
                return  # 终止生成器，无返回值

            callback = AsyncIteratorCallbackHandler()
            callbacks = [callback]

            # 获取 LLM
            llm = get_ChatOpenAI(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbacks,
            )

            context = "\n\n".join([doc["page_content"] for doc in docs])
            prompt_template = get_prompt_template("rag", prompt_name)  # 确保 prompt_name 总是有值
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages([i.to_msg_template() for i in history] + [input_msg])

            chain = chat_prompt | llm

            # 异步任务并发处理
            task = asyncio.create_task(wrap_done(chain.ainvoke({"context": context, "question": query}), callback.done))

            if len(source_documents) == 0:  # 没有找到相关文档
                source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

            if stream:
                # yield documents first
                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion.chunk",
                    content="",
                    role="assistant",
                    model=model,
                    docs=source_documents,
                )
                yield ret.model_dump_json()

                async for token in callback.aiter():
                    ret = OpenAIChatOutput(
                        id=f"chat{uuid.uuid4()}",
                        object="chat.completion.chunk",
                        content=token,
                        role="assistant",
                        model=model,
                    )
                    yield ret.model_dump_json()
            else:
                answer = ""
                async for token in callback.aiter():
                    answer += token
                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion",
                    content=answer,
                    role="assistant",
                    model=model,
                )
                yield ret.model_dump_json()

            await task
        except asyncio.exceptions.CancelledError:
            logger.warning("streaming progress has been interrupted by user.")
        except Exception as e:
            logger.error(f"error in knowledge chat: {e}")
            yield json.dumps({"error": str(e)})

    if stream:
        return EventSourceResponse(knowledge_base_chat_iterator())  # 流式返回
    else:
        return await knowledge_base_chat_iterator().__anext__()  # 返回第一个生成的结果
