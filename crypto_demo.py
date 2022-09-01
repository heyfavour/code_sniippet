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
from cryptography.hazmat.primitives.asymmetric import utils
from cryptography.exceptions import InvalidSignature

# len(IV) = 16
IV = "8b991e525526bc73"
PRI_KEY = """
-----BEGIN PRIVATE KEY-----
MIICdwIBADANBgkqhkiG9w0BAQEFAASCAmEwggJdAgEAAoGBAKHMbCrnTmHSnjvH
8P9vuy8vv5HNgZsbP6ur0dx823cnDEIjZgfsFaUeV/eaZXIi60IzI6eqigzA4LJX
RLkKG57biGQDvrnNJHuTTAkRSSbfwdxnpoQPWDcNHTUqGTcNYxYkB4YOYLOgVsmq
uFjq9C7hjIxjoSOwNwJ32p0nTssRAgMBAAECgYBhYd100TVPEWplhsjZpVEfbHi7
89nfFj5zP/4W3BtnktwA7rdZa1H3yNSfVZFbagL5HDiIkM94L5rOHFJjoN7QACt7
6PnJj7y2UZnPMtYGl9tPt++arHdEqsx2Z/NYr2WPzaAgPpvltWf1oC4EF1HWKWVo
kBUXgShykInusgpBtQJBANIFJSoiN5wd3YIVPv00kUs6Fq+wGnMG+ypLIAc1952v
bAg8VqSSTuT0/XqJGanxR0yGvoPaJpqvoF8WhQ2INxcCQQDFOKKoU/CYyXfEdjzF
3GJZN+o/3ltBEU5bNCcB14rH7rQC3ABz+adynHr9hOGr98O6xYbqTMb0mUCJasN6
I+gXAkEAr9ihAoM93phe9GEHqYhPMxaDEj04GCG7QPE/8umr1zqfENI8lXTvW+MJ
LYUHmPQth5S2hb2tXw04EQXRB8CKpwJBAMJ6C6NzSnBPUoPvmBQAMxcJVTvv1wp9
t0emUMS3OAnZL7cWHHhAecdB2OH/080Q//g/6b9HQHVYXdRj7CiYGbcCQDHNvzox
ddVJFwyMpMuHztqCGGqm6pKaCvBlpoirY3Cj0KLN+yQnz/c4jIgi0q463RbJPKnu
6TNr8rbLPX2of6U=
-----END PRIVATE KEY-----
"""
PUB_KEY = """
-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQChzGwq505h0p47x/D/b7svL7+R
zYGbGz+rq9HcfNt3JwxCI2YH7BWlHlf3mmVyIutCMyOnqooMwOCyV0S5Chue24hk
A765zSR7k0wJEUkm38HcZ6aED1g3DR01Khk3DWMWJAeGDmCzoFbJqrhY6vQu4YyM
Y6EjsDcCd9qdJ07LEQIDAQAB
-----END PUBLIC KEY-----
"""


class Crypto(object):
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with cls._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
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
        self.hash_type = hashes.SHA1()  # hashes.SHA256()

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

    def rsa_encrypt(self, aes_key: str):
        encrypted_data = self.PUB_KEY.encrypt(
            aes_key.encode(),
            asymmetric.padding.OAEP(
                mgf=asymmetric.padding.MGF1(algorithm=self.hash_type),
                algorithm=self.hash_type,
                label=None
            ),
        )
        # RSA加密出来是base64decode,需要转码
        return base64.b64encode(encrypted_data).decode()

    def rsa_decrypt(self, encrypted_data: Union[bytes, str]):
        if isinstance(encrypted_data, str): encrypted_data = base64.b64decode(encrypted_data)
        decrypted_data = self.PRI_KEY.decrypt(
            encrypted_data,
            asymmetric.padding.OAEP(
                mgf=asymmetric.padding.MGF1(algorithm=self.hash_type),
                algorithm=self.hash_type,
                label=None
            )
        )
        return decrypted_data.decode()

    def sign(self, data: Union[bytes, str]):
        if isinstance(data, str): data = bytes(data, encoding=self.CHARSET)
        signature = self.PRI_KEY.sign(
            data,
            asymmetric.padding.PSS(
                mgf=asymmetric.padding.MGF1(self.hash_type),
                salt_length=asymmetric.padding.PSS.MAX_LENGTH
            ),
            self.hash_type,
        )
        # return base64.b64encode(signature).decode()
        return signature

    def verify(self, data: Union[bytes, str], signature: [bytes, str]):
        if isinstance(data, str): data = bytes(data, encoding=self.CHARSET)
        if isinstance(signature, str): signature = base64.b64encode(signature)
        try:
            self.PUB_KEY.verify(
                signature,
                data,
                asymmetric.padding.PSS(
                    mgf=asymmetric.padding.MGF1(self.hash_type, ),
                    salt_length=asymmetric.padding.PSS.MAX_LENGTH
                ),
                self.hash_type,
            )
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


if __name__ == '__main__':
    import json

    data = json.dumps({"name": "test"})
    crypt = Crypto()
    aes_data, rsa_key, signature = crypt.encrypt(data)
    data, verify = crypt.decrypt(aes_data, rsa_key, signature)
    print(data,verify)
