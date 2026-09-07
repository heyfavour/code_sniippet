import pymongo

client = pymongo.MongoClient("mongodb://192.168.31.166:27017/")

try:
    client.admin.command("ping")
    print("ping success")
except Exception as exc:
    raise exc