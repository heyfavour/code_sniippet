"""
死信队列:超时 超长 拒绝
1:消息被消费端拒绝，使用 channel.basicNack 或 channel.basicReject ，并且此时requeue 属性被设置为false
2: 消息在队列的存活时间超过设置的TTL时间。
3:消息队列的消息数量已经超过最大队列长度，无法再继续新增消息到MQ中
4:一个队列中的消息的TTL对其他队列中同一条消息的TTL没有影响
"""
import time

import pika
import json, os
from multiprocessing import Process

HOST = "49.235.242.224"
PORT = 50009


def init():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()



    channel.exchange_declare(exchange='normal-ex', exchange_type='direct',durable=True)
    arguments = {}
    # 过期时间
    # arguments["x-message-ttl"] = 100 #可由生产者发送设置 两个一起设置由较小的决定
    arguments["x-dead-letter-exchange"] = "dead-ex"  # 死信交换机
    arguments["x-dead-letter-routing-key"] = ""  # 死信routing-key
    arguments["x-max-length"] = 100  # 死信routing-key
    channel.queue_declare(queue="normal-queue", durable=True, arguments=arguments)
    channel.queue_bind(queue="normal-queue", exchange='normal-ex', routing_key="")

    channel.exchange_declare(exchange='dead-ex', exchange_type='direct',durable=True)
    channel.queue_declare(queue="dead-queue", durable=True)
    channel.queue_bind(queue="dead-queue", exchange='dead-ex', routing_key="")
    channel.close()


def producer():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    channel.confirm_delivery()
    data = {"id": 1, "status": "成功"}
    try:
        for i in range(100):
            data["index"] = i
            channel.basic_publish(
                exchange='normal-ex',
                routing_key="",
                body=json.dumps(data, ensure_ascii=False),
                properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent, expiration="50000"),#50s
            )
    except Exception as e:
        print(f'ACK ERROR : {str(e)}')
    # 程序退出前，确保刷新网络缓冲以及消息发送给rabbitmq，需要关闭本次连接
    connection.close()
    print('[PRODUCER] SEND ALL.')


def comsumer_normal():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    channel.basic_qos(prefetch_count=1)  # 避免性能不同但是公平分发

    def callback(ch, method, properties, body):
        if json.loads(body)["index"] % 2 == 0:
            #ch.basic_nack(delivery_tag=method.delivery_tag, multiple=False, requeue=False)
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        else:
            print(f'[{os.getpid()}:NORMAL] Received :{json.loads(body)}')
            ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)

    channel.basic_consume(queue="normal-queue", on_message_callback=callback, auto_ack=False)
    print('[NORMAL] Waiting for message.')
    channel.start_consuming()


def comsumer_dead():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    channel.basic_qos(prefetch_count=1)  # 避免性能不同但是公平分发

    def callback(ch, method, properties, body):
        print(f'[{os.getpid()}:DEAD] Received :{json.loads(body)}')
        ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)

    channel.basic_consume(queue="dead-queue", on_message_callback=callback, auto_ack=False)
    print('[DEAD] Waiting for message.')
    channel.start_consuming()


if __name__ == '__main__':
    init()
    normal = Process(target=comsumer_normal)
    dead = Process(target=comsumer_dead)
    produce = Process(target=producer)
    produce.start()
    normal.start()
    dead.start()


