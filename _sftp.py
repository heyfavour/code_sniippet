import os
import stat
import paramiko
import logging

_SFTP = {
    "username": "xxxx",
    "password": "xxxx",
    "host": "xxxx",
    "port": "xxxx",
    "remote": {
        "get": {
            "file": "",
        },
        "put": {
            "file": "",
        },
    },
    "local": {
        "get": {
            "file": "",
        },
        "put": {
            "file": "",
        },
    }
}


class SFTP(object):
    def __init__(self, **kwargs):
        self.username = kwargs['username']
        self.password = kwargs['password']
        self.hostname = kwargs['host']
        self.port = kwargs.get('port', 22)

        self.show_process_section = 5  # 显示进度百分比

        self._channel_timeout = 60
        self.transport = None
        self.sftp = None
        self.default_buffer = kwargs.get('default_buffer', 16384)  # 每次读取大小
        # self.max_packet_size = kwargs.get('max_packet_size', 16384)  # 最大传输大小#
        self.window_size = kwargs.get('window_size', None)  #

    def connect(self, max_packet_size=16384):
        self.transport = paramiko.Transport((self.hostname, self.port))  ##创建一个ssh客户端client对象
        self.transport.connect(username=self.username, password=self.password)
        # max_packet_size=self.max_packet_size
        self.sftp = paramiko.SFTPClient.from_transport(self._transport,window_size=self.window_size)
        _channel = self.sftp.get_channel()
        _channel.settimeout(self._channel_timeout)

    def connectnopass(self):
        # self.private_key = '/root/.ssh/id_rsa'  # 本地密钥文件路径
        self.key = paramiko.RSAKey.from_private_key_file(self.private_key)

        self.ssh = paramiko.SSHClient()  ##创建一个ssh客户端client对象
        # 二选一
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  ##允许连接不在know_hosts文件中的主机
        # self.ssh.load_system_host_keys()  # 获取客户端host_keys
        self.ssh.connect(self.hostname, port=self.port, username=self.username, pkey=self.key)

        self._transport = self.ssh.get_transport()
        self.sftp = paramiko.SFTPClient.from_transport(self._transport)

    def close(self):
        if self._transport: self._transport.close()

    def process_bar(self, transferred, toBetransferred):
        if toBetransferred <= 0: return
        percents = round(100.00 * transferred / float(toBetransferred), 2)
        if percents % self.show_process_section == 0:
            LOGGER.info('传输进度:文件总大小【{}】,当前传输大小【{}】,传输进度【{}%】'
                        .format(toBetransferred, transferred, percents))

    def makedir(self, root, *sub_dirs):
        if not self.cli:
            self.connect()
        path = root
        for d in sub_dirs:
            if d not in self.cli.listdir(path):
                self.cli.mkdir(os.path.join(path, d))
            path = os.path.join(path, d)

    def is_exists(self, path):
        try:
            self.cli.stat(path)
            return True
        except IOError:
            return False

    def make_dirs(self, path):
        if not self.is_exists(path):
            dir_name, _ = os.path.split(path)
            self.make_dirs(dir_name)
            self.cli.mkdir(path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return

    # 递归遍历远程服务器指定目录下的所有文件
    def _get_all_files_in_remote_dir(self, sftp, remote_dir):
        all_files = list()
        files = sftp.listdir_attr(remote_dir)
        for file in files:
            filename = os.path.join(remote_dir, file.filename)
            if stat.S_ISDIR(file.st_mode):  # 如果是文件夹的话递归处理
                all_files.extend(self._get_all_files_in_remote_dir(sftp, filename))
            else:
                all_files.append(filename)
        return all_files

    def get_dir(self, remote_dir, local_dir):
        all_files = self._get_all_files_in_remote_dir(self.cli, remote_dir)
        for file in all_files:
            local_filename = file.replace(remote_dir, local_dir)
            local_filepath = os.path.dirname(local_filename)
            os.makedirs(local_filepath, exist_ok=True)
            self.cli.get(file, local_filename)

    def _get_all_files_in_local_dir(self, local_dir):
        all_files = list()
        for root, dirs, files in os.walk(local_dir, topdown=True):
            for file in files:
                filename = os.path.join(root, file)
                all_files.append(filename)
        return all_files

    def put_dir(self, local_dir, remote_dir):
        all_files = self._get_all_files_in_local_dir(local_dir)
        for file in all_files:
            remote_filename = file.replace(local_dir, remote_dir)
            remote_path = os.path.dirname(remote_filename)
            try:
                self.cli.stat(remote_path)
            except Exception:
                sub_dirs = [d for d in remote_path[len(remote_dir):].split('/') if d]
                self.makedir(remote_dir, *sub_dirs)
            self.cli.put(file, remote_filename)

    def put2(self, local_file_name, remote_file_name, callback=None):
        """
        断点上传
        :param callback:
        :param local_file_name:
        :param remote_file_name:
        :return:
        """
        LOGGER.info(f"{local_file_name} --> {remote_file_name} sftp远程传输 断点传输")
        size = 0
        stat = self.cli.stat(remote_file_name)
        with open(local_file_name, "rb") as f_local:
            f_local_size = os.path.getsize(local_file_name)
            f_local_size = f_local_size - self.cli.stat(remote_file_name).st_size  # 计算远程目录还需要上传的剩余大小，本地-远程的大小
            if f_local_size <= 0: return
            f_local.seek(stat.st_size)
            with self.cli.open(remote_file_name, "ab") as f_remote:
                tmp_buffer = f_local.read(self.default_buffer)
                while tmp_buffer:
                    size += len(tmp_buffer)
                    f_remote.write(tmp_buffer)
                    tmp_buffer = f_local.read(self.default_buffer)
                    if callback is not None:
                        callback(size, f_local_size)

    def get2(self, remote_file_name, local_file_name, callback=None):
        """
        断点下载
        :param callback:
        :param local_file_name:
        :param remote_file_name:
        :return:
        """
        LOGGER.info(f"{remote_file_name} --> {local_file_name} sftp远程传输 断点传输")
        size = 0
        stat = os.stat(local_file_name)
        with self.cli.open(remote_file_name, "rb") as f_remote:
            f_remote_size = self.cli.stat(remote_file_name).st_size
            f_remote_size = f_remote_size - stat.st_size  # 计算剩余上传大小,远程总大小-本地总大小
            if f_remote_size <= 0: return
            f_remote.seek(stat.st_size)
            with open(local_file_name, "ab") as f_local:
                tmp_buffer = f_remote.read(self.default_buffer)
                while tmp_buffer:
                    size += len(tmp_buffer)
                    f_local.write(tmp_buffer)
                    tmp_buffer = f_remote.read(self.default_buffer)
                    if callback is not None:
                        callback(size, f_remote_size)

    def get(self, remote_path, local_path, callback=None):
        """
        重写下载,支持断点下载
        :param remote_path:
        :param local_path:
        :param callback:
        :return:
        """
        # LOGGER.info(f"{remote_path} --> {local_path} sftp远程传输下载开始")
        if os.path.isfile(local_path):
            self.get2(remote_path, local_path, callback)
        else:
            self.cli.get(remote_path, local_path, callback)
        # LOGGER.info(f"{remote_path} --> {local_path} sftp远程传输下载结束")

    def put(self, local_path, remote_path, callback=None):
        """
        重写上传,支持断点上传
        """
        # LOGGER.info(f"{local_path} --> {remote_path} sftp远程传输上传开始")
        if os.path.basename(remote_path) in self.cli.listdir(os.path.dirname(remote_path)):
            self.put2(local_path, remote_path, callback)
        else:
            self.cli.put(local_path, remote_path, callback)
        # LOGGER.info(f"{local_path} --> {remote_path} sftp远程传输上传结束")


if __name__ == '__main__':
    import config

    with SFTP(**config.SFTP['MTSHF_WH']) as sftp:
        cli = sftp.connect()
