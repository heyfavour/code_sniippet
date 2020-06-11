# 简单邮件传输协议
import smtplib
import email
import time
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

# 设置邮箱的域名
HOST = 'smtp.qq.com'
# 设置邮件标题
SUBJECT = '这是一封测试邮件'
# 设置发件人邮箱
FROM = '619511821@qq.com'
# 设置收件人邮箱
TO = '1508822762@qq.com'	# 可以填写多个邮箱，用逗号分隔，后面会用split做逗号分割
message = MIMEMultipart('related')
# --------------------------------------发送文本-----------------
# 发送邮件正文到对方的邮箱中
message_html = MIMEText("TEST", 'plain', 'utf-8')	# \n为换行
message.attach(message_html)

# -------------------------------------添加文件---------------------
'''
message_xlsx = MIMEText(open('test.csv', 'rb').read(), 'base64', 'utf-8')
# 设置文件在附件当中的名字
message_xlsx['Content-Disposition'] = 'attachment;filename="test01.csv"'
message.attach(message_xlsx)
'''
# 设置邮件发件人
message['From'] = FROM
# 设置邮件收件人
message['To'] = TO
# 设置邮件标题
message['Subject'] = SUBJECT

# 获取简单邮件传输协议的证书
email_client = smtplib.SMTP_SSL()
# 设置发件人邮箱的域名和端口，端口为465
email_client.connect(HOST, '465')
# ---------------------------邮箱授权码------------------------------
result = email_client.login(FROM, 'nrdwppestgcobdde')
print('登录结果', result)
email_client.sendmail(from_addr=FROM, to_addrs=TO.split(','), msg=message.as_string())
# 关闭邮件发送客户端
email_client.close()