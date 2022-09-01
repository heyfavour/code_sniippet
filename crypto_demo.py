"""
1.电码本模式(Electronic Codebook Book(ECB))
    明文消息被分成固定大小的块（分组），并且每个块被单独加密
2.密码分组链接模式(Cipher Block Chaining(CBC))
    每一个分组要先和前一个分组加密后的数据进行XOR异或操作
    不利于并行计算  误差传递  初始化向量IV
3.计算器模式(Counter (CTR))

4.密码反馈模式(Cipher FeedBack (CFB))
    进阶版CBC
5.输出反馈模式(Output FeedBack (OFB))
    进阶版CFB

padding
RSA:
pkcs1(最基本)----pkcs5(对密钥加密)----pkcs8(在以上基础上安全存储移植等)
证书：
pkcs7(基本语法)----pkcs12(安全传输)
"""
import threading
import base64
import random

from typing import Union
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes, asymmetric
from cryptography.exceptions import InvalidSignature

# len(IV) = 16
IV = "8b991e525526bc73"
PRI_KEY = """
-----BEGIN PRIVATE KEY-----
MIICdgIBADANBgkqhkiG9w0BAQEFAASCAmAwggJcAgEAAoGBAMlLYcRK6Q0XMszy
GbNyiuGWxrfcXPUhUc5caRjMQF4hSf2uWdJ91xS9j3h3kg4ciYw53IvOu2MXh0/p
ycs3eVcoOv829X577r1eTZR6Z+3PM21ZH5LfJtuYE9BUW0kqR8VGCU/UjzaoRIoj
zm9bdt1vsFuBrrYK2AcQQgHgsxGLAgMBAAECgYBIPm7DRWNpGFdaKNXCiqx/lF6T
pFoUfDXhC1eI192OKwJkMov4OMPVpMb2JGvd9q4DDs0xvCuSv+IHc0/CSJGabFrK
RBSQMgfnduLSytIzHvrdmq4YN0txglP2JWulT4WrS7j5RGCNOSc0LkBQDpz+4Q7v
Bvzl5GU2CANKpeBUWQJBAOOSU6/w1E8H2GMJF90RDiIRH0pGKUveyje0W0O4Utzf
HN6QRblaB2RXq2hcwPQug9mE1R6yGPo9aQj2GQfZ2Z8CQQDicLhW04KVj3Kozttw
XgDZM/lXvfFN2JNPkuwLJHjzZjX/1V4dfs7ADSiu7BbKqbCrA8PhqkoBtrQ347uO
r5iVAkB2hwIbgx2xQ+7KNjQ9qeJoj+5yKvTbVWCRftiB/wD5lSNeMFqAXYm4E4lt
Q9Ij3A5EPtEZub0UqOOKDVOgKTEVAkEAur9dt/XN70yTslaPMVfFeVxc2hkDRkFE
FE9GLlZRDeOQy0IL0WWAW3E+ySxaC5/w3MlJJfZL/KfSb3l4eE+nFQJAOPAV2MPR
CT2KPWFXUYwQV6tgPYSqBpTJp5Averfobc2LqNgCUGwghJaB2/76pQISkYD/Emvb
9PLmxpoxxzT+nQ==
-----END PRIVATE KEY-----
"""
PUB_KEY = """
-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDJS2HESukNFzLM8hmzcorhlsa3
3Fz1IVHOXGkYzEBeIUn9rlnSfdcUvY94d5IOHImMOdyLzrtjF4dP6cnLN3lXKDr/
NvV+e+69Xk2UemftzzNtWR+S3ybbmBPQVFtJKkfFRglP1I82qESKI85vW3bdb7Bb
ga62CtgHEEIB4LMRiwIDAQAB
-----END PUBLIC KEY-----
"""


class Crypto(object):
    # _instance_lock = threading.Lock()
    #
    # def __new__(cls, *args, **kwargs):
    #     if not hasattr(cls, "_instance"):
    #         with cls._instance_lock:
    #             if not hasattr(cls, "_instance"):
    #                 cls._instance = super().__new__(cls)
    #     return cls._instance

    def __init__(self,
                 PRI_KEY=None,
                 PUB_KEY=None,
                 rsa_crypt_padding_type="PKCS1V15",
                 rsa_sign_padding_type="PKCS1V15",
                 sign_hash=hashes.SHA1(),
                 ):
        self.CHARSET = 'utf-8'
        self.iv = IV
        self.PRI_KEY = serialization.load_pem_private_key(PRI_KEY.encode(), password=None, backend=default_backend())
        self.PUB_KEY = serialization.load_pem_public_key(PUB_KEY.encode(), backend=default_backend())
        """
        with open("f{pri_key_path}}", "rb") as pri_key:
            self.PRI_KEY = serialization.load_pem_private_key(key_file.read(),password = None,backend = default_backend())
        with open("f{pub_key_path}}", "rb") as pub_key:
            self.PUB_KEY = serialization.load_pem_public_key(key_file.read(),backend = default_backend())
        """

        self.rsa_crypt_padding_type = rsa_crypt_padding_type
        self.rsa_sign_padding_type = rsa_sign_padding_type
        self.sign_hash = sign_hash

    @property
    def rsa_crypt_padding_dict(self):
        # PKCS1V15 = 固定位 + 随机数 + 明文消息
        # OAEP = 原文Hash + 随机数 + 分隔符 + 原文 #PKCS1V20
        _dict = {
            "PKCS1V15": asymmetric.padding.PKCS1v15(),
            "OAEP": asymmetric.padding.OAEP(
                mgf=asymmetric.padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            ),
        }
        return _dict

    @property
    def rsa_sign_padding_dict(self):
        _dict = {
            "PKCS1V15": asymmetric.padding.PKCS1v15(),
            "PSS": asymmetric.padding.PSS(
                mgf=asymmetric.padding.MGF1(hashes.SHA256()),
                salt_length=asymmetric.padding.PSS.MAX_LENGTH
            ),
        }
        return _dict

    def random_key(self):
        key = "".join([str(random.randint(0, 9)) for _ in range(16)])
        return key

    def aes_encrypt(self, data: str, key: str) -> str:
        # AES的要求的分块长度固定为128比特
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padding_data = padder.update(data.encode(self.CHARSET)) + padder.finalize()
        cipher = Cipher(algorithms.AES(key.encode()), modes.ECB(), backend=default_backend())
        # cipher = Cipher(algorithms.AES(key.encode()), modes.CBC(self.iv.encode()), backend=default_backend())
        encrypt_data = cipher.encryptor().update(padding_data)
        # return base64.b64encode(encrypt_data)#->bytes
        return str(base64.b64encode(encrypt_data), encoding=self.CHARSET)

    def aes_decrypt(self, data: Union[bytes, str], key: str) -> str:
        bytes_data = base64.b64decode(data)
        cipher = Cipher(algorithms.AES(key.encode()), modes.ECB(), backend=default_backend())
        # cipher = Cipher(algorithms.AES(key.encode()), modes.CBC(self.iv.encode()), backend=default_backend())
        unpdding_data = cipher.decryptor().update(bytes_data)
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        decrypt_data = unpadder.update(unpdding_data) + unpadder.finalize()
        return decrypt_data.decode(self.CHARSET)

    @property
    def _rsa_padding(self):
        return self.rsa_crypt_padding_dict[self.rsa_crypt_padding_type]

    def rsa_encrypt(self, aes_key: str):
        encrypted_data = self.PUB_KEY.encrypt(aes_key.encode(), self._rsa_padding)
        # RSA加密出来是base64decode,需要转码
        return base64.b64encode(encrypted_data).decode()

    def rsa_decrypt(self, encrypted_data: Union[bytes, str]):
        if isinstance(encrypted_data, str): encrypted_data = base64.b64decode(encrypted_data)
        decrypted_data = self.PRI_KEY.decrypt(encrypted_data, self._rsa_padding)
        return decrypted_data.decode()

    @property
    def _rsa_sign_padding(self):
        return self.rsa_sign_padding_dict[self.rsa_sign_padding_type]

    def sign(self, data: Union[bytes, str]):
        # 主流的RSA签名包括 RSA-PSS RSA-PKCS1v15
        # PSS更安全
        if isinstance(data, str): data = bytes(data, encoding=self.CHARSET)
        signature = self.PRI_KEY.sign(data, self._rsa_sign_padding, self.sign_hash)
        return base64.b64encode(signature).decode()

    def verify(self, data: Union[bytes, str], signature: [bytes, str]):
        if isinstance(data, str): data = bytes(data, encoding=self.CHARSET)
        if isinstance(signature, str): signature = base64.b64decode(signature.encode())
        try:
            self.PUB_KEY.verify(signature, data, self._rsa_sign_padding, self.sign_hash)
            return True
        except InvalidSignature:
            return False

    def encrypt(self, data):
        aes_key = self.random_key()  # random_keu
        aes_data = self.aes_encrypt(data, aes_key)  # AES data->aes_data
        # data = self.aes_decrypt(aes_data, aes_key)
        rsa_key = self.rsa_encrypt(aes_key)  # pub_prim SHA1withRSA aes_key->rsa_key
        signature = self.sign(data)
        return aes_data, rsa_key, signature

    def decrypt(self, aes_data, rsa_key, signature):
        key = self.rsa_decrypt(rsa_key)
        data = self.aes_decrypt(aes_data, key)
        verify = self.verify(data, signature)
        return data, verify

#改善 多渠道时 渠道单例
class Channel_A(Crypto):
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with cls._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        super().__init__(PRI_KEY, PUB_KEY, "PKCS1V15", "PKCS1V15")


if __name__ == '__main__':
    import json

    data = json.dumps({"name": "test"})
    crypt = Channel_A()
    aes_data, rsa_key, signature = crypt.encrypt(data)
    data, verify = crypt.decrypt(aes_data, rsa_key, signature)
    print(data, verify)
