# -*- coding: utf-8 -*-
import requests
import json
userAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36"
header = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Content-Length': '39',
    'Content-Type': 'application/json;charset=UTF-8',
    'Host': 'pedro.7yue.pro',
    'Pragma': 'no-cache',
    'Origin': 'http://face.cms.7yue.pro',
    "Referer": "http://face.cms.7yue.pro/",
    'User-Agent': userAgent,
}

def login(account, password):
    postUrl = "http://pedro.7yue.pro/cms/user/login"
    postData = {
        "username": account,
        "password": password,
    }
    responseRes = requests.post(postUrl, data = json.dumps(postData), headers = header)
    print(f"statusCode = {responseRes.status_code}")
    print(f"text = {responseRes.text}")

if __name__ == "__main__":
    login("root", "123456")

