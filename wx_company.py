"""
创办企业微信后，推送信息
"""
import json, requests

WECOM_ID = "xxxx"  # 企业ID
WECOM_SECRET = "xxxx"
WECOM_AID = "xxxx"  # 应用ID


def send_to_wecom(text, wecom_cid, wecom_secret, wecom_aid, wecom_touid='@all'):
    get_token_url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={wecom_cid}&corpsecret={wecom_secret}"
    response = requests.get(get_token_url).content
    access_token = json.loads(response).get('access_token')
    if access_token and len(access_token) > 0:
        send_msg_url = f'https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}'
        data = {"touser": wecom_touid, "agentid": wecom_aid, "msgtype": "text", "text": {"content": text},
                "duplicate_check_interval": 600
                }
        response = requests.post(send_msg_url, data=json.dumps(data)).content
        return response
    else:
        return False


if __name__ == '__main__':
    data = {
        "text": "推送的信息",
        "wecom_cid": WECOM_ID,
        "wecom_secret": WECOM_SECRET,
        "wecom_aid": WECOM_AID,
    }
    send_to_wecom(**data)
