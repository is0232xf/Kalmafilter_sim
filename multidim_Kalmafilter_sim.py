# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 21:11:16 2018

@author: FujiiChang
"""
import math
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

theta = 35.140845 # 緯度 
earth_R = 6378137 # 地球の半径．WGS84の値を用いる
dy = 1/(360/2*math.pi*earth_R) # 緯度単位距離. 1mあたりの度数
dx = 1/((2*math.pi*earth_R*math.cos(theta*math.pi/180))/360) # 経度単位距離. 1mあたりの度数

# 関数
def model(x, A, B, u):
    return np.dot(A, x) + np.dot(B, u)

def true(x):
    noise_y = random.normal(0.0, 0.5*dy) # 緯度方向の移動に関する外乱があった時の誤差 ←実験をして設定をし直す必要がある
    noise_x = random.normal(0.0, 0.5*dx) # 経度方向の移動に関する外乱があった時の誤差 ←実験をして設定をし直す必要がある
    noise = np.array([[noise_y],
                      [noise_x]])
    return x + noise

# 実際はこの関数のところにGPS受信機で得た値をとる
def observe(x):
    noise_y = random.normal(0.0, 1*dy) # 緯度方向の観測に関する外乱があった時の誤差 ←実験をして設定をし直す必要がある
    noise_x = random.normal(0.0, 1*dx) # 経度方向の観測に関する外乱があった時の誤差 ←実験をして設定をし直す必要がある
    noise = np.array([[noise_y],
                      [noise_x]])
    return x + noise

def system(x, A, B, u):
    true_val = true(model(x, A, B, u))
    obs_val = observe(true_val)
    return true_val, obs_val

def Kalman_Filter(m, V, y, A, B, u, Q, R):
    # 予測
    m_est = model(m, A, B, u)
    V_est = np.dot(np.dot(A, V), A.transpose()) + Q
    
    # 観測更新
    K = np.dot(V_est, np.linalg.inv(V_est + R))
    m_next = m_est + np.dot(K, (y - m_est))
    V_next = np.dot((np.identity(2) - K), V_est)
    
    return m_next, V_next

if __name__ == "__main__":
    
    # セットアップ
    itr = 30 
    direction = -10 # システムが進む方向．単位は度
    u_original = 0.2 # 任意の方向の速度, 単位はm．秒速にして秒速0.2mとしている
    uy = dy*math.sin(direction/180*math.pi) # 緯度方向の速度, 単位は度．秒速にして秒速0.2mとしている
    ux = dx*math.cos(direction/180*math.pi) # 経度方向の速度, 単位は度．秒速にして秒速0.2mとしている
    u = np.array([[uy],
                 [ux]])
    q_y = dy # 緯度方向のシステム誤差の分散
    q_x = dx # 経度方向のシステム誤差の分散
    r_y = 10*dy # 緯度方向の観測誤差の分散．距離にして10mの観測誤差があると仮定している
    r_x = 10*dx # 経度方向の観測誤差の分散．距離にして10mの観測誤差があると仮定している
    
    A = np.identity(2) # 状態方程式のA
    B = np.ones((1, 2)) # 状態方程式のB
    x = np.array([[35.140957],
                  [135.982505]]) # 真値
    m = np.array([[35.140957],
                  [135.982505]]) # 推定値
    V = np.identity(2) # 推定値の初期共分散行列(勝手に設定して良い)
    Q = np.array([[q_y, 0],
                  [0, q_x]]) # システム誤差の共分散行列
    R = np.array([[r_y, 0],
                  [0, r_x]]) # 観測誤差の共分散行列

    # 記録用
    rec = np.empty((6, itr))

    # main loop
    for i in range(itr):
        x, y = system(x, A, B, u)
        m, V = Kalman_Filter(m, V, y, A, B, u, Q, R)
        rec[0, i] = x[0]
        rec[1, i] = x[1]
        rec[2, i] = m[0]
        rec[3, i] = m[1]
        rec[4, i] = y[0]
        rec[5, i] = y[1]

    # 描画
    plt.plot(rec[0, :], rec[1, :], color="blue", marker="o", label="true")
    plt.plot(rec[2, :], rec[3, :], color="red", marker="^", label="estimated")
    plt.plot(rec[4, :], rec[5, :], color="green", marker="+", label="observed")
    plt.legend(loc='lower right')
    plt.show()
    print("start position: " + str(rec[0,0]) + "," + str(rec[1,0]))
    print("goal position: " + str(rec[0,29]) + "," + str(rec[1,29]))