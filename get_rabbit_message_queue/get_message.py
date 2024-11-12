import pika

# 连接到 RabbitMQ 服务器
credentials = pika.PlainCredentials('admin', 'yikdata.com')
connection_params = pika.ConnectionParameters('8.154.45.125', 5672, '/', credentials)
connection = pika.BlockingConnection(connection_params)
channel = connection.channel()

# 声明队列，确保队列存在
channel.queue_declare(queue='test_queue')

# 定义回调函数，处理接收到的消息
def callback(ch, method, properties, body):
    print(f"Received: {body.decode()}")  # 解析消息内容

# 告诉 RabbitMQ 使用上面的回调函数来处理从 'test_queue' 中接收到的消息
channel.basic_consume(queue='test_queue', on_message_callback=callback, auto_ack=True)

print('Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
