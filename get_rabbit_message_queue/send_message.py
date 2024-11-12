import pika
import json
rabbitmq_host = '8.154.45.125'
rabbitmq_username = 'admin'
rabbitmq_password = 'yikdata.com'
queue_name = 'test_queue'


# 连接到 RabbitMQ 服务器
credentials = pika.PlainCredentials(rabbitmq_username, rabbitmq_password)
connection_params = pika.ConnectionParameters(rabbitmq_host, 5672, '/', credentials)
connection = pika.BlockingConnection(connection_params)
channel = connection.channel()

# 连接到 RabbitMQ 服务器

# connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
# channel = connection.channel()

# 声明队列
channel.queue_declare(queue='test_queue')

# 发送消息
# for i in range(2):
#     if i == 0:
#         message = {"image_url": "/root/autodl-tmp/temp_image.jpg"}
#     else:
message = {"image_url": "https://jsbqtest-new.oss-cn-hangzhou.aliyuncs.com/apiRegistration/202108Z11MY00013098/0/931c1b6c423246a29a1b1799befdb054.jpg"}
channel.basic_publish(exchange='', routing_key='test_queue', body=json.dumps(message))
print(f"Sent: {message}")
# message = 'Hello, RabbitMQ!'
# channel.basic_publish(exchange='', routing_key='test_queue', body=message)
# print(f"Sent: {message}")

# 关闭连接
connection.close()
