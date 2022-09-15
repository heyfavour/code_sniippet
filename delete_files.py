import os
import re
import datetime
import paramiko


def name_pattern(name):
    name = name.replace("%Y", '[0-9]{4}').replace("%m", '[0-9]{1,2}').replace("%d", "[0-9]{1,2}")
    return "^" + name + "$"


def get_delete_files(path, name, run_date, cycle):
    fileNames = os.listdir(path)
    result = []
    pattern = re.compile(name_pattern(name))
    for file_name in fileNames:
        # 正则判断
        if pattern.match(file_name):
            result.append(os.path.join(path, file_name))
    if len(result) < cycle:
        result = []
    else:
        result.sort()
        result = result[:len(result) - cycle]
    return result


def clear_local(path, name, run_date, cycle):
    if int(run_date) > int(datetime.datetime.now().strftime("%Y%m%d")): raise Exception("日期异常")
    delete_files = get_delete_files(path, name, run_date, cycle)
    for file in delete_files: os.remove(file)


def clear_remote(path, name, run_date, cycle, remote):
    if int(run_date) > int(datetime.datetime.now().strftime("%Y%m%d")): raise Exception("日期异常")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(remote["ip"], username=remote["username"], password=remote["password"])
    cmd_to_execute = f"ls {path}"
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd_to_execute)
    pattern = re.compile(name_pattern(name))
    result = []
    for line in ssh_stdout.readlines():
        file_name = line.replace("\n", "")
        if pattern.match(file_name):
            result.append(os.path.join(path, file_name))
    if len(result) < cycle:
        result = []
    else:
        result.sort()
        result = result[:len(result) - cycle]
    cmd_to_execute = f"rm {' '.join(result)}"
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd_to_execute)
    ssh.close()


def clear(path, name, run_date, cycle, remote=None):
    if remote:
        clear_remote(path, name, run_date, cycle, remote)
    else:
        clear_local(path, name, run_date, cycle)


def run_clear(run_date):
    # DEMO
    # clear("/app/log","xxx.log.%Y-%m-%d",run_date,10)
    # clear("/app/log","xxx.log.%Y-%m-%d",run_date,60,{"ip":"x.x.x.x","username":"xxx","password":"xxx"})
    pass


if __name__ == "__main__":
    run_date = "20220915"
    run_clear(run_date)
