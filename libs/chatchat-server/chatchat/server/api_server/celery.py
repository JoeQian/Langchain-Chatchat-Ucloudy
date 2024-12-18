from celery import Celery

# 使用 RabbitMQ 作为消息队列
celery_app = Celery(
    "chat_completions",
    broker="pyamqp://guest:guest@localhost//",  # 这里是 RabbitMQ 的默认设置
    backend="rpc://",  # 可以使用 RPC 作为结果后端，或者其他后端
)

# 配置 Celery
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",  # 设置时区
)
