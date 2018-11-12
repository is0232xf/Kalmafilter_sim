# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 17:13:03 2018

@author: FujiiChang
"""

import numpy as np
import numpy.random as random

# ↓ここの各パラメータの値を別ファイルから読み込めるようにすると汎用性が上がる
ux = 0 # ロール方向の角速度, 単位は度/単位時間
uy = 0 # ピッチ方向の角速度, 単位は度/単位時間
uz = 0 # よー方向の角速度, 単位は度/単位時間
u = np.array([[ux],
              [uy],
              [uz]])
q_x = 0.1 # ロール方向のシステム誤差の分散
q_y = 0.1 # ピッチ方向のシステム誤差の分散
q_z = 0.1 # ヨー方向のシステム誤差の分散
r_x = 0.1 # ロール方向の観測誤差の分散．
r_y = 0.1 # ピッチ方向の観測誤差の分散．
r_z = 0.1 # ヨー方向の観測誤差の分散．

A = np.identity(3) # 状態方程式のA
B = np.ones((1, 3)) # 状態方程式のB
x = np.array([[0.0],
              [0.0],
              [0.0]]) # 真値
m = np.array([[0.0],
              [0.0],
              [0.0]]) # 推定値
V = np.identity(3) # 推定値の初期共分散行列(勝手に設定して良い)
Q = np.array([[q_x, 0, 0],
              [0, q_y, 0],
              [0, 0, q_z]]) # システム誤差の共分散行列
R = np.array([[r_x, 0, 0],
              [0, r_y, 0],
              [0, 0, r_z]]) # 観測誤差の共分散行列

# 関数
def model(x, A, B, u):
    return np.dot(A, x) + np.dot(B, u)

def true(x):
    noise_x = random.normal(0.0, 0.1) # ロール方向のノイズ
    noise_y = random.normal(0.0, 0.1) # ピッチ方向のノイズ
    noise_z = random.normal(0.0, 0.1) # ヨー方向のノイズ
    noise = np.array([[noise_x],
                      [noise_y],
                      [noise_z]])
    return x + noise

# 実際はこの関数のところにGPS受信機で得た値をとる
def observe(x):
    noise_x = random.normal(0.0, 0.1)
    noise_y = random.normal(0.0, 0.1)
    noise_z = random.normal(0.0, 0.1)
    noise = np.array([[noise_x],
                      [noise_y],
                      [noise_z],])
    return x + noise

def system(x, A, B, u):
    true_val = true(model(x, A, B, u))
    obs_val = observe(true_val)
    return true_val, obs_val

def Kalman_Filter(m, V, y, A, B, u, Q, R):
    # 予測
    m_est = model(m, A, B, u)
    print("m_est:\n", m_est)
    print("y    :\n", y)
    V_est = np.dot(np.dot(A, V), A.transpose()) + Q
    
    # 観測更新
    K = np.dot(V_est, np.linalg.inv(V_est + R))
    m_next = m_est + np.dot(K, (y - m_est))
    print("m_next:\n", m_next)
    V_next = np.dot((np.identity(3) - K), V_est)
    
    return m_next, V_next
