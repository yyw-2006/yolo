"""
模块名称：app
作用：图片审核 MVP 的应用包（FastAPI 入口、配置、管线与模型适配）。
使用方法：
  - 启动：uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  - 调用：POST /audit/image 上传图片文件
"""

