import datetime
import sys
line = ["0", "测试", "20221004", "XXXXXXX", "XXXXXXX", "100", "100", "100", "100", "0", "0", "0", "2", "TEST1"]
data = [line for i in range(1000000)]


# 0:00:01.361810
def wirte1():
    with open("tmp.txt", "w+") as f:
        for i, line in enumerate(data):
            line[0] = str(i)
            line = ",".join(line) + "\n"
            f.write(line)


def wirte2():
    lines = []
    for i, line in enumerate(data):
        line[0] = str(i)
        line = ",".join(line)
        lines.append(line)
    print(sys.getsizeof(lines)/1048/1046)#M
    open("tmp.txt", "w+").write("\n".join(lines))


if __name__ == '__main__':
    start = datetime.datetime.now()
    wirte1()  # 0:00:01.423579
    #wirte2()  # 0:00:01.036075
    end = datetime.datetime.now()
    print(end - start)
