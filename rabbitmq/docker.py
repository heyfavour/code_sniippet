# 新建用户
"""
docker search rabbitmq
docker pull rabbitmq
docker run -d --hostname my-rabbit --name rabbit -p 15672:15672 -p 5673:5672 rabbitmq # -p 外网端口：docker的内部端口
docker ps -a
docker exec -it 容器id /bin/bash
rabbitmq-plugins enable rabbitmq_management
http://ip:15672

rabbitmqctl add_user admin admin
rabbitmqctl set_user_tags username admin
"""
# 角色
"""
RabbitMQ用户角色分为Administrator、Monitoring、Policymaker、Management、Impersonator、None共六种角色。
(1)Administrator:超级管理员，可登陆管理控制台(启用management plugin的情况下)，可查看所有的信息，并且可以对用户，策略(policy)进行操作。
(2)Monitoring:监控者，可登陆管理控制台(启用management plugin的情况下)，同时可以查看rabbitmq节点的相关信息(进程数，内存使用情况，磁盘使用情况等)。
(3)Policymaker:策略制定者，可登陆管理控制台(启用management plugin的情况下)，同时可以对policy进行管理。但无法查看节点的相关信息。
(4)Management:普通管理者，仅可登陆管理控制台(启用management plugin的情况下)，无法看到节点信息，也无法对策略进行管理。
(5)Impersonator:模拟者，无法登录管理控制台。
(6)None:其他用户，无法登陆管理控制台，通常就是普通的生产者和消费者。
"""
# user
"""
查看当前所有用户：rabbitmqctl list_users
添加用户：rabbitmqctl add_user username passwd
赋予其administrator角色:rabbitmqctl set_user_tags username administrator
删除角色:rabbitmqctl delete_user username
"""
# vhost
"""
列出所有虚拟主机:rabbitmqctl list_vhosts
创建虚拟主机:rabbitmqctl add_vhost [vhostpath]
列出虚拟主机上所有角色对应权限:rabbitmqctl list_permissions -p [vhostpath]
删除虚拟主机:rabbitmqctl delete_vhost [vhostpath]
"""
# 权限
"""
# 对何virtual hosts具有配置、写、读的权限通过正则表达式来匹配，具体命令如下：rabbitmqctl set_permissions -p <vhostpath> <user> <conf> <write> <read>
#“---” 双引号中的信息为正则表达式
#".*" 表示配置任何队列和交换机
#"checks-.*"表示只能配置名字以"checks-"开头的队列和交换机
#" " 不匹配队列和交换机
设置权限:rabbitmqctl set_permissions -p / username ".*" ".*" ".*"
查看用户的权限:rabbitmqctl list_user_permissions username
清除用户对某个virtual hosts的权限:rabbitmqctl clear_permissions -p / username
"""
##
"""
rabbitmqctl stop_app
rabbitmqctl reset
rabbitmqctl stop
"""

# DEMO
"""
rabbitmqctl add_vhost /product

rabbitmqctl add_user product product
rabbitmqctl set_permissions -p / product " " " " " "
rabbitmqctl set_permissions -p /product product ".*" ".*" ".*"

rabbitmqctl set_user_tags product None
"""