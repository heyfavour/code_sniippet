"""
    可用的代理IP池子
"""
import requests
from bs4 import BeautifulSoup


def web_site_jiangxianli(page_num):
    ip_list = []

    url = f"https://ip.jiangxianli.com/?page={page_num}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    add_flag = False
    count = 0
    for link in soup.find_all("link"):
        ip = link["href"].replace(r"//", "")
        if ip == "github.com":
            add_flag = True
            continue
        if add_flag and count <= 14:
            ip_list.append(ip)
            count = count + 1
    print(ip_list)
    return ip_list


def get_ip_proxies():
    ip_list = []
    for page_num in range(1, 4):
        uncheck_list = web_site_jiangxianli(page_num)
        ip_list.extend(uncheck_list)
    return ip_list


def check_ip_proxis(ip_list):
    valid_ip_list = []

    for proxy_ip in ip_list:
        proxy = {'http': f'http://{proxy_ip}'}
        try:
            url = "http://www.baidu.com"
            requests.get(url, proxies=proxy, timeout=1)
        except Exception as exc:
            continue
        valid_ip_list.append(proxy_ip)
    return valid_ip_list


if __name__ == '__main__':
    ip_list = get_ip_proxies()
    ip_list = check_ip_proxis(ip_list)
    print(ip_list)
