"""
1bit-0  41bit-时间戳 10bit 机器id 12-bit序列号
41bit 可以记录69年 设置好起始时间，比如今年是 2022 ，那么可以用到 2091 年
雪花算法存在的问题
1.时钟偏斜问题（Clock Skew）
当获取系统时间，机器时间回拨，则可能会造成ID重复。我们知道普通的计算机系统时钟并不能保证长久的一致性，可能发生时钟回拨等问题，这就会导致时间戳不准确，进而产生重复ID。

2.时间数据位数限制
Snowfake的方案由于时间数据位数的限制，存在与2038年问题相似的理论极限。虽然目前的系统设计考虑数十年后的问题还太早，但是还是有可能存在的。

3.机器id上限
如果机器过多，可能会导致10bit的工作机器id不够用。
"""
import time
import logging

log = logging.getLogger(__name__)
# 64 位 id 的划分,通常机器位和数据位各为 5 位
DATACENTER_BITS = 5  # 机房id
WORKER_BITS = 5  # 计算机id
SEQUENCE_BITS = 12  # 序列号长度
# 最大取值计算,计算机中负数表示为他的补码
WORKER_LIMIT = -1 ^ (-1 << WORKER_BITS)  # 31 11111 2**5 - 1
DATACENTER_LIMIT = -1 ^ (-1 << DATACENTER_BITS)  # 31 11111 2**5 - 1
# 移位偏移计算
WORKER_SHIFT = SEQUENCE_BITS  # 12
DATACENTER_SHIFT = SEQUENCE_BITS + WORKER_BITS  # 17=12+5
TIMESTAMP_LEFT_SHIFT = SEQUENCE_BITS + WORKER_BITS + DATACENTER_BITS  # 12+5+5 = 22
# 序号循环掩码
SEQUENCE_MASK = -1 ^ (-1 << SEQUENCE_BITS)  # 4095 = 2**12-1

# Twitter 元年时间戳
TWEPOCH = 1577808000000  # 2020-01-01


class SnowFlake(object):
    def __init__(self, datacenter_id, worker_id, sequence=0):
        """
        :param datacenter_id: 机房id
        :param worker_id: 机器id
        :param sequence: 序列码
        return 18 id
        """
        if datacenter_id > WORKER_LIMIT or datacenter_id < 0: raise ValueError('datacenter_id 值越界')
        if worker_id > DATACENTER_LIMIT or worker_id < 0: raise ValueError('worker_id 值越界')

        self.worker_id = worker_id
        self.datacenter_id = datacenter_id
        self.sequence = sequence
        self.last_timestamp = -1  # 上次计算的时间戳

    def _timestamp(self):  # time.time 返回当前时间的时间戳(从1970纪元开始的浮点秒数)
        return int(time.time() * 1000)

    def generate_id(self):
        timestamp = self._timestamp()  # 获取当前时间戳
        # 时钟回拨
        if timestamp < self.last_timestamp:
            log.warning(f'clock is moving backwards.  waiting until {self.last_timestamp}')
            timestamp = self.wait_millis(self.last_timestamp)
        if timestamp == self.last_timestamp:  # 同一毫秒的处理
            self.sequence = (self.sequence + 1) & SEQUENCE_MASK
            # 一毫秒内超过4096笔业务
            if self.sequence == 0: timestamp = self.wait_millis(self.last_timestamp)
        else:
            self.sequence = 0

        self.last_timestamp = timestamp
        time_zone = (timestamp - TWEPOCH) << TIMESTAMP_LEFT_SHIFT
        datacentor_zone = self.datacenter_id << DATACENTER_SHIFT
        worker_zone = self.worker_id << WORKER_SHIFT
        _id = time_zone | datacentor_zone | worker_zone | self.sequence
        return _id

    def wait_millis(self, last_timestamp):
        timestamp = self._timestamp()
        while timestamp <= last_timestamp:
            time.sleep((last_timestamp - timestamp) / 1000)
            timestamp = self._timestamp()
        return timestamp


if __name__ == '__main__':
    worker = SnowFlake(1, 2)
    for i in range(200): print(worker.generate_id())
