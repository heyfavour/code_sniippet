from gmssl import sm4
import base64
import binascii
from gmssl import sm2, func


class Sm4Cryption():
    def __init__(self):
        self.key = b'3l5butlj26hvv313'
        self.iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'  # bytes类型
        self.crypt_sm4 = sm4.CryptSM4()

    # ECB 相同输入产生相同输出 可并行计算
    def encrypt_ecb(self, value):
        value = self.check_value(value)
        self.crypt_sm4.set_key(self.key, sm4.SM4_ENCRYPT)
        encrypt_value = self.crypt_sm4.crypt_ecb(value)  # bytes类型
        return encrypt_value

    def decrypt_ecb(self, encrypt_value):
        self.crypt_sm4.set_key(self.key, sm4.SM4_DECRYPT)
        decrypt_value = self.crypt_sm4.crypt_ecb(encrypt_value)  # bytes类型
        return decrypt_value

    # CBC 密码分组链接 安全性高于ECB
    def encrypt_cbc(self, value):
        value = self.check_value(value)
        self.crypt_sm4.set_key(self.key, sm4.SM4_ENCRYPT)
        encrypt_value = self.crypt_sm4.crypt_cbc(self.iv, value)  # bytes类型
        return encrypt_value

    def decrypt_cbc(self, encrypt_value):
        self.crypt_sm4.set_key(self.key, sm4.SM4_DECRYPT)
        decrypt_value = self.crypt_sm4.crypt_cbc(self.iv, encrypt_value)  # bytes类型
        return decrypt_value

    def check_value(self, string_value):  # 加密时 value 需为bytes类型
        if isinstance(string_value, bytes):
            return string_value
        elif isinstance(string_value, str):
            return string_value.encode("utf-8")
        else:
            raise Exception("please input string or bytes")


class Sm2Cryption():
    def __init__(self, ):
        # 16进制的公钥和私钥
        self.private_key = '00B9AB0B828FF68872F21A837FC303668428DEA11DCD1B24429D0C99E24EED83D5'
        self.public_key = 'B9C9A6E04E9C91F7BA880429273747D7EF5DDEB0BB2FF6317EB00BEF331A83081A6994B8993F3F5D6EADDDB81872266C87C018FB4162F5AF347B483E24620207'
        self.sm2_crypt = sm2.CryptSM2(public_key=self.public_key, private_key=self.private_key)

    def encrypt(self, data):  ##数据和加密后数据为bytes类型
        data = self.check_value(data)
        enc_data = self.sm2_crypt.encrypt(data)
        return enc_data

    def decrypt(self, enc_data):
        enc_data = self.check_value(enc_data)
        dec_data = self.sm2_crypt.decrypt(enc_data)
        return dec_data.decode("utf-8")

    def sign(self, data):
        data = self.check_value(data)
        random_hex_str = func.random_hex(self.sm2_crypt.para_len)
        sign = self.sm2_crypt.sign(data, random_hex_str)  # 16进制
        return sign

    def verify(self, sign, data):
        data = self.check_value(data)
        return self.sm2_crypt.verify(sign, data)  # 16进制

    def check_value(self, string_value):  # 加密时 value 需为bytes类型
        if isinstance(string_value, bytes):
            return string_value
        elif isinstance(string_value, str):
            return string_value.encode("utf-8")
        else:
            raise Exception("please input string or bytes")
    # 已舍弃
    # def sign_with_sm3(self, data):
    #     data = self.check_value(data)
    #     sign = self.sm2_crypt.sign_with_sm3(data) #  16进制
    #     return sign
    #
    # def verify_with_sm3(self, sign, data):
    #     data = self.check_value(data)
    #     return self.sm2_crypt.verify_with_sm3(sign, data)


if __name__ == '__main__':
    value = "你好"
    crypt = Sm2Cryption()
    enc_data = crypt.encrypt(value)
    dec_data = crypt.decrypt(enc_data)
    print(enc_data, dec_data)
    sign = crypt.sign(value)
    verify = crypt.verify(sign, value)

    value = "你好"
    crypt = Sm4Cryption()
    # ECB
    encrypt_value = crypt.encrypt_ecb(value)
    decrypt_value = crypt.decrypt_ecb(encrypt_value)
    # CBC
    encrypt_value = crypt.encrypt_cbc(value)
    decrypt_value = crypt.decrypt_cbc(encrypt_value)
