import configparser
import os.path

project_root = 'D:\\PyDocument\\detect_vpn_modle1' 


def getConfig(section,key=None):
    config = configparser.ConfigParser()  #初始化一个configparser类对象
    file_path = os.path.join(project_root, 'config.ini')
    config.read(file_path,encoding='utf-8') #读取config.ini文件内容
    if key!=None:
        return config.get(section,key)  #获取某个section下面的某个key的值
    else:
        return config.items(section)  #或者某个section下面的所有值


# a = getConfig('URL')
# print(a)

