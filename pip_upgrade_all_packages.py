import os
#p = os.system("pip list") 无返回值
p = os.popen("pip list")
#pip list --outdated 也可以选择更新过期库
modules = p.readlines()[2:]
for m in modules:
    module = m.split()[0]
    update_module = "pip install --upgrade {module}".format(module = module)
    update_result = os.system(update_module)
    if update_result != 0:raise Exception(update_module,'failed')


