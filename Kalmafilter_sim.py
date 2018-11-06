# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:31:41 2018

@author: FujiiChang
"""

import matplotlib.pyplot as plt 
import random

def make_figure(x, y, x_hat):
    plt.plot(x, "-o")
    plt.plot(y, "-x")
    plt.plot(x_hat, "-+")
    plt.show()

# セットアップ

# 状態と誤差分散の初期値設定
x_zero = 0
p_zero = 0

# 正規分布のパラメータ
#  mはmeanの意味．つまり平均. varはvarianceの意味．つまり分散.
m_v = 0
m_w = 0
var_v = 1
var_w = 2

# 記憶領域
x = []
x_hat = []
y = []
p = []
p_bar = []
g = []
e_x_y = []
e_x_x_hat = []

# カウンタ変数
k = 0
flag = 0 
itr = 100

if __name__ == "__main__":
    
    x.append(0)
    x_hat.append(x_zero)    
    p.append(p_zero)
    p_bar.append(0)
    g.append(0)
    while(1):
        print("k: ", k)
        print("------------------------------------------")
        # 真値の設定と出力の設定
        v = random.gauss(m_v, var_v)
        w = random.gauss(m_w, var_w)
        y.append(x[k] + w)
        if k != itr-1:
            x.append(x[k] + v)
        # 真の値と観測値を表示
        print("x    : ", x[k])
        print("y    : ", y[k])
        if flag == 0:
            flag = 1
        else:
            # 予測ステップ
            x_hat_bar = x_hat[k-1]
            p_bar.append(p[k-1] + var_v)
            # フィルタリングステップ
            g.append(p_bar[k]/(p_bar[k]+var_w))
            x_hat.append(x_hat_bar+g[k]*(y[k]-x_hat[k-1]))
            p.append((1-g[k])*p_bar[k])
            
        # 推定値の表示
        print("x_hat: ", x_hat[k])
        print("##########################################")
        k+=1
        # 終了させる
        if k == itr:
            break
    make_figure(x, y, x_hat)