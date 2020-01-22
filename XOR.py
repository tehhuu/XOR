import numpy as np
from numpy import exp
from fractions import Fraction
import math
#np.random.seed(0)

def sigmoid(a):
    a = -a
    g = 1 / (1 + np.exp(a.astype(float)))
    return g


w_1 = np.random.randn(2,2) #中間層への重み
b_1 = np.random.randn(1,2) #中間層へのバイアス
w_2 = np.random.randn(1,2) #出力層への重み
b_2 = np.random.randn(1) #出力層へのバイアス

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) #入力
t = np.array([[0], [1], [1], [0]]) #正解出力

lr = 0.01 #学習率
epochs = 1000000 #エポック数

for _ in range(epochs):
    sum_dldw2 =  0  
    sum_dldb2 =  0
    sum_dldw1 =  0
    sum_dldb1 =  0

    for i in range(4):  #4パターンの入力について重み、バイアスの更新分を計算
        ###順伝播
        u1 = np.dot(x[i], w_1) + b_1            
        h = sigmoid(u1)
        u2 = np.dot(h, w_2.reshape(2,1)) + b_2
        y = sigmoid(u2)            

        ###逆伝播
        dldy =  -2 * (t[i] - y)
        dydu2 = y * (1 - y)
        du2dw2 = h

        dldu2 = dldy * dydu2
        dldw2 = dldu2 * du2dw2
        dldb2 = dldu2 * 1

        du2dh = w_2
        dhdu1 = h * (1 - h)
        du1dw1 = x[i]

        dldw1 = np.dot(du1dw1[:,None],(dhdu1 * du2dh * dldu2)[None,:])
        dldb1 = dhdu1 * du2dh * dldu2

        #各入力パターンに対する重み・バイアスの更新分を、計算し終わるごとに足していく
        sum_dldw2 +=  dldw2[0]
        sum_dldb2 +=  dldb2[0]
        sum_dldw1 +=  dldw1.reshape((2,2))
        sum_dldb1 +=  dldb1[0]

    #重み・バイアスを更新
    w_2 = w_2 - lr * sum_dldw2 * Fraction(1, 4) 
    b_2 = b_2 - lr * sum_dldb2 * Fraction(1, 4)
    w_1 = w_1 - lr * sum_dldw1 * Fraction(1, 4)
    b_1 = b_1 - lr * sum_dldb1 * Fraction(1, 4)

#訓練後の出力を確認
u1 = np.dot(x, w_1) + b_1
h = sigmoid(u1)
u2 = np.dot(h, w_2.reshape(2,1)) + b_2
y = sigmoid(u2)
for i in range(len(y)):
    print('{0} → {1}'.format(x[i], y[i]))

