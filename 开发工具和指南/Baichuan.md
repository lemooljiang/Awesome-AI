# <center>Baichuan</center>

## 下载与资源
[官网 |](https://www.baichuan-ai.com/home#introduce)
[文档 |](https://platform.baichuan-ai.com/docs/api#4)


## python使用
```py
import requests
from dotenv import dotenv_values
import json
import time
import hashlib
import sys

url = "https://api.baichuan-ai.com/v1/chat"
env_vars = dotenv_values('.env')
baichuan_api_key = env_vars['BAICHUAN_API_KEY']
baichuan_secret_key = env_vars['BAICHUAN_SECRET_KEY']

def calculate_md5(input_string):
    md5 = hashlib.md5()
    md5.update(input_string.encode('utf-8'))
    encrypted = md5.hexdigest()
    return encrypted

def baichuanChat(query):
    print("666 响应header:")
    data = {
        "model": "Baichuan2-53B",
        "messages": query
    }

    json_data = json.dumps(data)
    time_stamp = int(time.time())
    signature = calculate_md5(baichuan_secret_key + json_data + str(time_stamp))

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + baichuan_api_key,
        "X-BC-Request-Id": "your requestId",
        "X-BC-Timestamp": str(time_stamp),
        "X-BC-Signature": signature,
        "X-BC-Sign-Algo": "MD5",
    }

    response = requests.post(url, data=json_data, headers=headers)

    if response.status_code == 200:
        result = json.loads(response.text)
        # print(339, result["data"]["messages"][0]["content"], file=sys.stderr)
        return result["data"]["messages"][0]["content"]

    else:
        # print("请求失败，状态码:", response.status_code)
        return 'error', 500
```