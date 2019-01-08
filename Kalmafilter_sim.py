# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:31:41 2018

@author: FujiiChang
"""

import matplotlib.pyplot as plt
import random

# 折れ線グラフの表示
def make_figure(x, y, x_hat):
    plt.plot(x, "-+", label="groud truth", color="blue")
    plt.plot(y, "-+", label="observed data", color="red")
    plt.plot(x_hat, "-+", label="estimated data", color="green")
    plt.xlabel("step")
    leg = plt.legend()
    leg.get_frame().set_alpha(0.5) # 範例を透明にする
    plt.show()

# セットアップ

# 状態と誤差分散の初期値設定
x_zero = 0
p_zero = 0

# 正規分布のパラメータ
#  mはmean（平均）の意味. varはvariance（分散）の意味．
m_u = 0
m_w = 0
var_u = 10
var_w = 3

# 各ステップにおける各値を保持するリスト
x = [] # 位置の真の値
x_hat = [] # 位置の推定値
y = [] # 位置の観測値
p = [] # 事後分散
p_bar = [] # 事前分散
g = [] # カルマンゲイン

# カウンタ変数
k = 0
# 条件分岐変数
flag = 0
# 試行回数
itr = 50

# 累積誤差
e_x_y = 0
e_x_x_hat = 0

if __name__ == "__main__":

    x.append(0)
    x_hat.append(x_zero)
    p.append(p_zero)
    p_bar.append(0)
    g.append(0)
    while(1):
        print("k: ", k)
        print("------------------------------------------")
        # 真値と観測値の設定
        u = random.gauss(m_u, var_u)
        w = random.gauss(m_w, var_w)
        y.append(x[k] + w)
        if k != itr-1:
            x.append(x[k] + u)
        # 真値と観測値を表示
        print("x    : ", x[k])
        print("y    : ", y[k])
        if flag == 0:
            flag = 1
        else:
            # 予測ステップ
            u = random.gauss(m_u, var_u)
            x_hat_bar = x_hat[k-1]+u
            p_bar.append(p[k-1] + var_u)
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
    # 累積誤差を計算する
    for i in range(itr):
        e_x_y += abs(x[i]-y[i])**2
        e_x_x_hat += abs(x[i]-x_hat[i])
    e_x_y = (e_x_y)/i
    e_x_x_hat = (e_x_x_hat)/i

    # 結果の表示
    make_figure(x, y, x_hat)
    print("MSE(y): ", e_x_y)
    print("MSE(x_hat): ", e_x_x_hat)
