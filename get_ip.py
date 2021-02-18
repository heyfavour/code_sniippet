import requests


def get_ip_by_api(ip):
    resp = requests.get(url=f'http://ip-api.com/json/{ip}')
    data = resp.json()
    print(data)


if __name__ == '__main__':
    ip = "13.212.9.169"
    get_ip_by_api(ip)
