from logging import exception
from logging.config import valid_ident
import os
import glob
from sre_constants import CATEGORY
import subprocess as sp
from time import sleep
from tkinter import EW
from typing import Type
from dataset_group import *  # noqa
from numpy import append, ma
from scapy.all import PcapReader
from scapy.all import Ether, IP, TCP, UDP, DNS, Padding
import numpy as np
import matplotlib.pyplot as plt
import binascii
import pickle
import traceback
import myUtils as mu

pcap_max_num = int(mu.getConfig('parameter', 'pcap_max_num'))  # 每一类pcap会话文件夹使用最大数量
frames_min_threshold = int(mu.getConfig('parameter', 'frames_min_threshold'))  # 一个有效的会话中必须包含此数值以上的帧
frames_max_threshold = int(mu.getConfig('parameter', 'frames_max_threshold'))  # 一个有效的会话中最多包含此数值以上的帧
frame_max_length = int(mu.getConfig('parameter', 'frame_max_length'))  # 流的最长字节数，大于截断，小于补0


flow_category = mu.getConfig('parameter', 'flow_category')
flow_category = flow_category.split(",")
flow_class = len(flow_category)


hex_total_list = []
for _ in range(flow_class * 2):
    hex_total_list.append([])


dirname = mu.getConfig('URI', '2_SplitDirName') 

flow_pickle_save_dir = mu.getConfig('URI', 'test_pickle_save_dir') 


def start_deal():
    for root, dirs, _ in os.walk(dirname):
        task_num = len(dirs)
        for dir in dirs:
           # 同类型流量文件的会话汇总列表
            dir_route = os.path.join(root, dir)
            print("now deal:====",dir,"====还剩余:" + str(task_num))
            task_num -= 1
            deal_pcap_files(dir_route, dir)  
    try:
        for i in range(flow_class):
            filename = flow_category[i] + '_vflow.pkl'
            vpn_pickle_save_route = os.path.join(flow_pickle_save_dir, str(frames_min_threshold) +'_' + str(frames_max_threshold) + filename)
            with open(vpn_pickle_save_route,'wb') as f:
                pickle.dump(hex_total_list[i],f)
        for i in range(flow_class):
            filename = flow_category[i] + '_nvflow.pkl'
            i += flow_class  
            novpn_pickle_save_route = os.path.join(flow_pickle_save_dir, str(frames_min_threshold) +'_' + str(frames_max_threshold) + filename)
            with open(novpn_pickle_save_route,'wb') as f:
                pickle.dump(hex_total_list[i],f)
    except Exception:
        print(traceback.format_exc())



# 处理文件夹所属标签，对文件大小进行排序
def deal_pcap_files(dir_route, dir):
    for in_root, _, pcap_files in os.walk(dir_route):
        if dir in TYPES:
            try:        
                label, category_label, _ = TYPES[dir] # 分类
                pcap_file_size_list = []  # 文件及其大小
                for pcap_file in pcap_files:
                    ab_pcap_file = os.path.join(in_root, pcap_file)
                    pcap_file_size_list.append([ab_pcap_file,os.path.getsize(ab_pcap_file)])
                pcap_file_size_list.sort(key=lambda obj: -obj[1]) # 从大至小排列
                file_idc = 0
                # sessions_list = [] 
                for item in pcap_file_size_list:
                    pcap_file = item[0]
                    session_list = [] # 一个会话所记录的数据 
                    res = deal_pcap_frame(pcap_file, session_list)
                    if res:
                        save_category(session_list, category_label,label)
                        # sessions_list.append(session_list)
                        file_idc += 1
                        if file_idc < pcap_max_num:
                            pass
                        else:
                            break
                        # session_hex_list.append(session_list) # , vpn_label, category_label, app_label
            except Exception as ex:
                print("处理pcap文件出错",str(ex))
                print(traceback.format_exc())
                    

def save_category(session_list, category_label,label):
    if label:
        hex_total_list[category_label - 1].append(session_list)
    else:
        hex_total_list[category_label + 5].append(session_list)

# 处理单个会话文件
def deal_pcap_frame(pcap_file_route, session_list):
    frame_idc = 0   
    try:
        for packet in PcapReader(pcap_file_route):
            if IP in packet:  # 消除可能造成过拟合的参数
                packet['IP'].dst = "0"
                packet['IP'].src = "0"
            if Ether in packet: 
                raw_content = bytes(packet.payload)
            else:
                raw_content = bytes(packet)
            if len(raw_content) < frame_max_length:
                raw_content += b'\0' * (frame_max_length - len(raw_content))
            raw_content = raw_content[:frame_max_length]
            # raw_hex = binascii.hexlify(raw_content)
            data = np.frombuffer(raw_content, dtype=np.uint8, count=frame_max_length) /255 # 16进制转换10进制
            session_list.append(data)
            frame_idc += 1
            if frame_idc < frames_max_threshold:
                pass
            else:
                return True
        if frame_idc < frames_min_threshold:
            return False
        else:
            zero_need = frames_max_threshold - frame_idc
            for _ in range(zero_need):
                add_ = np.zeros((frame_max_length), dtype=np.uint8)
                session_list.append(add_)
            return True
    except Exception as ex:
        print("处理帧数据出错",str(ex))    
        print(traceback.format_exc())
        return False


# # 是否需要忽略该包
# def omit_packet(packet):
#     # SYN, ACK or FIN flags set to 1 and no payload
#     if TCP in packet and (packet.flags & 0x13):
#         # not payload or contains only padding
#         layers = packet[TCP].payload.layers()
#         if not layers or (Padding in layers and len(layers) == 1):
#             return True
#     # DNS segment
#     if DNS in packet:
#         return True

#     return False


def main():
    start_deal()


main()