"""
sqlalchemy 对象缓存机制
__dict__ -> _sa_instance_state -> sqlalchemy.orm.state.InstanceState ->  session 、orm mapper
"""
db, Object = None, None
from sqlalchemy.orm import sessionmaker
object = Object.query.get(1)
# time.sleep
object = Object.query.get(1)

# sqlalchemy 会缓存object 如果time.sleep期间，数据库发生了变化，并不会更新到object

db.expire(object)  # 清除缓存的对象 下次使用会重新查询一次
db.refresh(object)  # 清楚对象缓存的数据 并立即触发一次查询  session.refresh(object, ['status']) ->只更新特性属性 避免数据量过大
db.expire_all()  # 清楚所有缓存的数据

db.flush()  # 所有本地修改写入数据库 但不提交
db.commit()  # 提交数据
"""
mysql  默认隔离级别是可重复读(RR)因此需要设置为读已提交(RC)上诉才有效 [ru 读未提交|RC 读已提交|RR 重复读|串行化]
oracle 默认RC
db2 默认cs [ur 读未提交|cs 读取行加锁|rs结果集加锁|rr可重复读]
"""
db.expunge(object)#释放实例  SQLAlchemy有个特点，当你的session会话结束以后，它会销毁你插入的这种临时数据，你再想访问这个data就访问不了了。所以我们可以释放这个数据
