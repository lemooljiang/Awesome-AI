# <center> DeepTutor </center>

## 下载与资源
[github |](https://github.com/HKUDS/DeepTutor)
[文档 |](https://docs.deeptutor.info/zh-cn/explore/)
[介绍 |](https://mp.weixin.qq.com/s/ETwQtU6Zw_ciP78aD4R8IQ)


## 简介
香港大学开源的智能辅导系统，由香港大学 Data Intelligence Lab 开发，团队成员在 AI 教育领域有深厚积累。项目从 2025 年 12 月发布。六大学习模式共享上下文、TutorBot 自主导师、Book Engine 活书编译器，让 AI 不只是聊天工具，而是真正懂你的个人导师。


## 安装
PyPI安装是最顺滑的路径。需要 Python 3.11+ 以及 PATH 上的 Node.js 20+。
```py
mkdir -p deeptutor && cd deeptutor
virtualenv deep_env
source deep_env/bin/activate
# source deep_env/Scripts/activate
pip install deeptutor
# pip install deeptutor -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install -U deeptutor
deeptutor init     # 提示输入端口 + LLM 提供商 + 可选 embedding
deeptutor start    # 同时启动后端 + 前端；保持终端开着

deeptutor init 会询问：
    后端端口（默认 8001）
    前端端口（默认 3782）
    LLM 提供商 / base URL / API key / 模型  
    可选 embedding 提供商（用于 RAG / 知识库）
    openai https://api.openai.com/v1 
    openai text-embedding-3-large	3072
deeptutor start 之后，打开终端里打印的前端地址 —— 默认是 http://127.0.0.1:3782。按 Ctrl+C 同时停掉后端和前端。
    跳过 deeptutor init 也行，应用会以默认端口和空配置启动，进 设置 → LLM 再配也可以。

工作区位于运行命令时所在的目录（首次启动会在那里创建 data/ 目录）。覆盖位置：
DEEPTUTOR_HOME=/opt/deeptutor deeptutor start
deeptutor start --home /opt/deeptutor
```

## 热重载
后端支持 --reload。在两个独立终端里分别启后端和前端，开发体验最好：
```py
# 终端 A —— 带热重载的后端
python -m uvicorn deeptutor.api.main:app \
  --host 0.0.0.0 --port 8001 \
  --reload --reload-exclude "web/*" --reload-exclude "data/*"

# 终端 B —— 前端 dev server
cd web && npm run dev -- -p 3782
```

## Nginx反向代理
如果在服务器端启动服务，可以简单地由Nginx反向代理来提供https服务。
```
server { 
	server_name  example.com;

	gzip on;
	gzip_buffers 4 16k;
	gzip_comp_level 6;
  gzip_vary on;
  gzip_types text/plain text/css application/json application/x-javascript text/xml application/xml application/xml+rss text/javascript;
  # 限制上传文件大小 如果DeepTutor需要上传大文件或模型 建议调大
  client_max_body_size 50m;

	location / {
      proxy_pass http://127.0.0.1:3782;
      proxy_set_header   Host             $host;
      proxy_set_header   X-Real-IP        $remote_addr;
      proxy_set_header   X-Forwarded-For  $proxy_add_x_forwarded_for;
      proxy_http_version 1.1;

      # 【核心】支持 WebSocket 连接 防止AI对话时流式传输或长连接中断
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";

      # 超时设置（防止大模型生成回答时间过长导致 Nginx 报 504 超时）
      proxy_read_timeout 600s;
      proxy_send_timeout 600s;
    }

	error_page   500 502 503 504  /50x.html;
	location = /50x.html {
		root   /usr/share/nginx/html;
	}
 ....
}
```