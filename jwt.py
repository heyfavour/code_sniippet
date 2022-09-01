# python-jose
from datetime import datetime, timedelta
from jose import jwt

ALGORITHM = "HS256"
SECRET_KEY = "VlBIHQ6005uORLSmnivHPHe2-8Xu6wwCD75JpvOUUqI"

"""
iss: jwt的签发者
sub: jwt面向的用户
aud: 接收jwt的一方
exp: 过期时间
iat: 签发时间
"""


def gen_token(user_id):
    expire = datetime.utcnow() + timedelta(minutes=60 * 24)
    to_encode = {"exp": expire, "sub": str(user_id)}
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token


def decode_jwt(token):
    options = {"verify_exp": True}
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options=options)
    except jwt.ExpiredSignatureError:
        # TODO 提示前端重新登录
        print("token过期")
    except jwt.JWTError:
        # TODO 无效签名
        print("token过期")


if __name__ == '__main__':
    token = gen_token("wang")
    data = decode_jwt(token)
