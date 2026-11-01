# 导入 os 模块，该模块提供了与操作系统进行交互的功能，包括文件和目录操作
import os

# os.path.abspath(__file__) 用于获取当前脚本文件的绝对路径
# __file__ 是 Python 内置变量，表示当前正在执行的脚本文件的路径
# 将获取到的绝对路径赋值给变量 pwd
pwd = os.path.abspath(__file__)

# os.path.dirname() 函数用于获取指定路径的父目录
# 这里对 pwd 调用该函数，得到当前脚本文件所在目录的路径，并更新 pwd 的值
pwd = os.path.dirname(pwd)

# 再次调用 os.path.dirname() 函数，获取上一次得到的目录的父目录路径
# 最终 pwd 变量存储的是当前脚本文件所在目录的上一级目录的路径
pwd = os.path.dirname(pwd)