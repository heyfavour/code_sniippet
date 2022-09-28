import time

import pika
import uuid

"""
Property

Delivery mode: 是否持久化，1 - Non-persistent，2 - Persistent
Headers：{'x-delay': data["delay-time"]}
Properties: You can set other message properties here (delivery mode and headers are pulled out as the most common cases). Invalid properties will be ignored. Valid properties are:
content_type ： 消息内容的类型  text/json
content_encoding： 消息内容的编码格式
priority： 消息的优先级
correlation_id：关联id
reply_to: 用于指定回复的队列的名称
expiration： 消息的失效时间
message_id： 消息id
timestamp：消息的时间戳
type： 类型
user_id: 用户id
app_id： 应用程序id
cluster_id: 集群id
Payload: 消息内容(必须是以字符串形式传入)
"""
HOST = "49.235.242.224"
PORT = 50009


def client():
    credentials = pika.PlainCredentials("admin", "admin")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()

    queue = channel.queue_declare(queue='', exclusive=True)
    client_queue = queue.method.queue
    response = None
    corr_id = str(uuid.uuid1())

    def callback(ch, method, properties, body):
        print(f'[Client] callback receive {body}')
        nonlocal response
        if corr_id == properties.correlation_id:
            response = body

    channel.basic_consume(queue=client_queue, on_message_callback=callback, auto_ack=True)
    channel.basic_publish(
        exchange='',
        routing_key='rpc_queue',
        properties=pika.BasicProperties(reply_to=client_queue, correlation_id=corr_id),
        body=str(30)
    )
    while response is None:
        connection.process_data_events()
    print(f'[Client] Receive {response}')


def server():
    credentials = pika.PlainCredentials("admin", "admin")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()

    queue = channel.queue_declare(queue='rpc_queue', exclusive=True)

    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, props, body):
        print(f"[SERVER] receive {props.reply_to} {props.correlation_id}")
        n = int(body)
        response = int(n) + 1
        ch.basic_publish(
            exchange='',  # 使用默认交换机
            routing_key=props.reply_to,  # response发送到该queue
            properties=pika.BasicProperties(correlation_id=props.correlation_id),  # 使用correlation_id让此response与请求消息对应起来
            body=str(response)
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue='rpc_queue', on_message_callback=callback, auto_ack=False)

    print(" [x] Awaiting RPC requests")
    channel.start_consuming()


if __name__ == '__main__':
    from multiprocessing import Process

    s = Process(target=server)
    c = Process(target=client)
    s.start()
    c.start()
