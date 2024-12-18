from fastapi import APIRouter
from chatchat.server.utils import BaseResponse, ListResponse

from  chatchat.server.admin_server.server  import controller_list
# 创建 admin_router
admin_router = APIRouter(prefix="/admin", tags=["Admin Management"])

# 示例接口：获取管理员列表
@admin_router.get("/list_admins", response_model=ListResponse, summary="获取管理员列表")
async def list_admins():
    return {"data": [{"id": 1, "username": "admin1"}, {"id": 2, "username": "admin2"}]}

# 示例接口：创建管理员
@admin_router.post("/create_admin", response_model=BaseResponse, summary="创建管理员")
async def create_admin(username: str, password: str):
    return {"message": f"Admin {username} created successfully"}

# 包含 admin_router 到 kb_router 或直接注册到主路由
from fastapi import FastAPI

app = FastAPI()

# 注册新的 admin 路由
app.include_router(admin_router)

for controller in controller_list:
    app.include_router(router=controller.get('router'), tags=controller.get('tags'))