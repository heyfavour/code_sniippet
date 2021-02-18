# !/usr/bin/python
# -*- coding: utf-8 -*-
from ftplib import FTP
import os


class MyFTP():
    def __init__(self, host, user_name, password):
        self.host = host
        self.user_name = user_name
        self.password = password
        self.localpath = os.getcwd()
        self.ftp = self.__init_ftp(host, user_name, password)

    def __init_ftp(self, host, user_name, password):
        ftp = FTP()
        ftp.set_debuglevel(2)  # 打开调试级别2，显示详细信息
        ftp.connect(host, 21)  # 连接
        ftp.login(user_name, password)  # 登录，如果匿名登录则用空串代替即可
        return ftp

    def downloadfile(self, remotepath, localpath):
        bufsize = 1024  # 设置缓冲块大小
        self.ftp = open(localpath, 'wb')  # 以写模式在本地打开文件
        self.ftp.retrbinary('RETR ' + remotepath, self.ftp.write, bufsize)  # 接收服务器上文件并写入本地文件
        self.ftp.set_debuglevel(0)  # 关闭调试
        self.ftp.close()  # 关闭文件

    def uploadfile(ftp, remotepath, localpath):
        bufsize = 1024
        fp = open(localpath, 'rb')
        ftp.storbinary('STOR ' + remotepath, fp, bufsize)  # 上传文件
        ftp.set_debuglevel(0)
        fp.close()

    def quit(self):
        self.ftp.quit()


if __name__ == "__main__":
    ftp = MyFTP("104.160.32.20", "root", "wzx940516")
    #ftp.downloadfile("***", "***")
    #ftp.quit()
