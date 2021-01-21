# -*- coding: utf-8 -*-

import os
import os.path


def dfs_showdir(path, depth):
    if depth == 0:
        print("root:[" + path + "]")
    for item in os.listdir(path):
        if not os.path.isdir(os.path.join(path,item)): continue
        #if "frontend" in item or "__pycache__" in item or ".idea" in item:continue
        if '.git' not in item:
            print("|      " * depth + "|--" + item)
            newitem = os.path.join(path,item)
            if os.path.isdir(newitem):
                dfs_showdir(newitem, depth + 1)


if __name__ == '__main__':
    path = r'C:\Users\Mr.Wang\Desktop\myfile\vue-element-admin-fastapi'
    dfs_showdir(path, 0)

