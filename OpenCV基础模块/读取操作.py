# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:34:55 2026

@author: 徐逸东
"""

import cv2 as cv

# 读取图像
img = cv.imread("./image/nature_image.jpg")
# 如果图像路径错误或文件不存在，cv.imread() 会返回 None
if img is None:
    # 打印错误信息
    print("Error: Could not load image.")
    # 退出程序
    exit()
    
# "Display window" 是显示窗口的名称，可以自定义
# img 是要显示的图像数据
cv.imshow("Display window", img)

# 等待按键输入
# 参数 0 表示无限等待，直到用户按下任意键
# 返回值 k 是用户按下的键的 ASCII 码值
k = cv.waitKey(0)

# 检查用户是否按下 Esc 键（ASCII 码为 27）
if k == 27:
    # 关闭所有 OpenCV 窗口
    cv.destroyAllWindows()