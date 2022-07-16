# # # # # # # #
#@Author      : YuLin
#@Date        : 2022-07-08 19:56:40
#@LastEditors : YuLin
#@LastEditTime: 2022-07-08 19:59:15
#@Description : 文件描述
# # # # # # # #
import argparse

parser = argparse.ArgumentParser()
    
# 给parser实例添加属性
parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
parser.add_argument('-bs', type=int, default=128, help='batch size for dataloader')
parser.add_argument('-epoches', type=int, default=15, help='batch size for dataloader')

# 把刚才的属性给args实例，后面就可以直接使用
args = parser.parse_args()

print(args.gpu)