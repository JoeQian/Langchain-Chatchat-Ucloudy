from celery import Celery
from chatchat.server.chat.chat import chat
from chatchat.server.chat.kb_chat import kb_chat
from chatchat.server.chat.feedback import chat_feedback
from chatchat.server.chat.file_chat import file_chat
from chatchat.server.db.repository import add_message_to_db
from chatchat.server.utils import get_OpenAIClient, get_tool, get_tool_config
from chatchat.settings import Settings
from chatchat.utils import build_logger
from .celery import celery_app  # 导入 Celery 实例

logger = build_logger()

@celery_app.task(bind=True)
async def process_chat_completions(self, body, conversation_id, message_id):
    """
    处理聊天请求（异步任务）
    """
    client = get_OpenAIClient(model_name=body.model, is_async=True)
    extra = {**body.model_extra} or {}
    for key in list(extra):
        delattr(body, key)

    # check tools & tool_choice in request body
    if isinstance(body.tool_choice, str):
        if t := get_tool(body.tool_choice):
            body.tool_choice = {"function": {"name": t.name}, "type": "function"}
    if isinstance(body.tools, list):
        for i in range(len(body.tools)):
            if isinstance(body.tools[i], str):
                if t := get_tool(body.tools[i]):
                    body.tools[i] = {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.args,
                        },
                    }

    # Handle the tool_choice or agent chat
    if body.tool_choice:
        tool = get_tool(body.tool_choice["function"]["name"])
        if not body.tools:
            body.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.args,
                    },
                }
            ]
        if tool_input := extra.get("tool_input"):
            try:
                add_message_to_db(
                    chat_type="tool_call",
                    query=body.messages[-1]["content"],
                    conversation_id=conversation_id,
                )
            except Exception as e:
                logger.warning(f"failed to add message to db: {e}")
            tool_result = await tool.ainvoke(tool_input)
            # Handle the tool result and send response
            return tool_result

    # 如果是 agent chat
    if body.tools:
        result = await chat(
            query=body.messages[-1]["content"],
            metadata=extra.get("metadata", {}),
            conversation_id=conversation_id,
            message_id=message_id,
            history_len=-1,
            history=body.messages[:-1],
            stream=body.stream,
            chat_model_config=extra.get("chat_model_config", {}),
            tool_config=extra.get("tool_config", {}),
            max_tokens=body.max_tokens,
        )
        return result

    # LLM chat
    try:
        result = await client.chat.completions.create(body)
        return result
    except Exception as e:
        logger.error(f"Error processing chat completions: {e}")
        raise self.retry(exc=e)
