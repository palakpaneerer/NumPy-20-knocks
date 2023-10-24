# これはNumPyフレームワークの20本ノックです。
# 以下のURLの問題について記載しております。
# https://www.youtube.com/watch?v=k4YzlaOXfvQ

#　インポート、warnings非表示
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Q1. ベクトルの定義
a = np.array([1,2,3,4]) #方向性無し
b = np.array([[4,-1,6]]) #方向性も定義 縦ベクトル


# Q2. 行列の定義
C = np.array(
    [[3,-2],
     [7,1]])
D = np.array(
    [[3,-2,0,1],
     [7,1,-1,2],
     [4,-5,1,3]])


# Q3.　ベクトルの作成
a = np.zeros((1,2)) #タプルで指定する
b = np.ones((0,4))


# Q4. 単位行列の作成
C = np.eye(3)
D = np.eye(3,4) 


# Q5. 基本情報の確認
print("形状", a.shape)
print("次元", a.ndim)
print("データ型", a.dtype)
print("要素数", a.size)


# Q6. インデックスとスライシング 要素の取得
a = np.arange(1,13,2) #1~13で2個飛ばし。最後の数字は含まない。
B = np.array(
    [[2,4,6],
     [-1,-2,-3],
     [0,-2,3]])

print("aの3個目~最後", a[2:])
print("aの3個目~最後から一つ目", a[2:-1])
print("aの逆順", a[::-1])

print("Bの1行目", B[0])
print("Bの3行目1列目(0)", B[2,0])
print("Bの2~3行目2~3列目(-2,-3,-2,3)", B[1:,1:])


# Q7. 四則演算
A = np.array(
    [[1,3],
    [-2,4]])
B = np.array(
    [[2,-1],
    [3,0]])


print("足し算", A+B)
print("行列積", A@B)
print("アダマール積（同じ場所の掛け算）", A*B)


# Q8. 転置
print(a.T) #2次元又は方向性を決まったベクトルで無いと転置効かない


# Q9. 行列式,　逆行列(行列と逆行列を掛けると谷行列になる。)
det = np.linalg.det(A)

det == 0 #行列式が0ではないので逆行列が存在する

Ainv = np.linalg.inv(A)

print("単位行列なってか", A @ Ainv)


# Q10.　ブロードキャスト（ベクトルや行列にスカラーを掛ける）
A = np.array(
    [[1,3],
    [-2,4]])

broadcast = A * 5


# Q11. ベクトルのサイズ変更
a = np.arange(12)
print(a)
a = a.reshape(3,4)
print(a)


# Q12. 統計値確認
B = np.array(
    [[2,4,6],
     [-1,5,3],
     [0,-2,3]])
print(B.min(),B.max(), B.sum(),B.mean(),B.var(),B.std())
print(B.min(axis=1)) #直観とは逆。横方向で夫々minの数字を探す。


# Q13.　ユニバーサル関数（Numpyで用意されている関数のこと）
# 平方根、eの指数関数（ネイピア数）
print("平方根", np.sqrt(B))
print("eの指数関数", np.exp(B))


# Q14. Part2 sin, cos
print("平方根", np.sin(B))
print("eの指数関数", np.cos(B))


# Q15. 行列の結合 縦方向に結合, 横方向に結合
A = np.array(
    [[0,1,-1],
     [2,4,-3],
     [5,-2,7]])
B = np.array(
    [[1,2,3],
     [2,4,-3],
    [5,-2,7]])

C = np.vstack((A,B))
D = np.hstack((A,B))


# Q16. 行列の分解 縦方向に分解、横方向に分解
C = np.vsplit(C, 2)
print("元のAと同じか", C[0])
D = np.hsplit(D,2)
print("元のAと同じか", D[0])


# Q17. グラフ化 
# 0~2piまでの500サンプル、sin波、cos波をプロット
import matplotlib.pyplot as plt
data = np.linspace(0, 2*np.pi, 500)
data1= np.sin(data)
data2= np.cos(data)
plt.plot(data1)
plt.plot(data2)
plt.show()

# 0~30でからランダム100個抽出し、ヒストグラム
data = np.random.randint(0, 31, 100)
plt.hist(data, bins=10)
plt.show()


# Q18. 演習　AとBを行列式で計算　C=AB　
A = np.array(
    [[0,1,-1],
     [2,4,-3],
     [5,-2,7],
     [5,5,5]])
B = np.array(
    [[1,2,3],
     [2,4,-3],
    [5,-2,7]])

print(A.shape, B.shape) #AとBを掛けられるか確認
C = A@B
print(C)


# Q19. 演習　Cの2行2列目までをDに保存。その後転置して、逆行列を求める。
D = C[:2,:2]
Dt = D.T
Dtinv = np.linalg.inv(Dt)
print("演習", D, Dt, Dtinv)


# Q20. 演習　-100~100までの乱数300サンプル 2つ作成、　それぞれx,y軸の値として散布図作成
data1 = np.random.randint(-100,101,300)
data2 = np.random.randint(-100,101,300)
plt.scatter(x=data1, y=data2)




