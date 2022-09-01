# pip install passlib
# pip instal bcrypt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
HASH_PASSWORD = "$2b$12$ePjKQ0qivxagZxz9GtNbtOtRH4LG8yrQnwkw/Bzt2.CTcTAE9D8Vy"


def verify_password(password):
    _verify = pwd_context.verify(password, HASH_PASSWORD)
    return _verify


def change_password(old, new):
    if not pwd_context.verify(old, HASH_PASSWORD): raise Exception("密码验证失败")
    hashed_password = pwd_context.hash(new)
    return hashed_password


if __name__ == '__main__':
    password = "123456789"
    new = "987654321"
    print(verify_password(password))
    print(change_password(password, new))
