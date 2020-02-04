# coding: utf-8

import numpy as np # 配列
import time # 時間
from matplotlib import pyplot as plt # グラフ
import os # フォルダを作成可能（意外と便利）

# chainer
from chainer import Variable, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


# y=sin(x)のデータセットをN個分作成
def get_data(N, Nte):
    x = np.linspace(0, 2 * np.pi, N+Nte)
    # 学習データとテストデータに分ける
    ram = np.random.permutation(N+Nte)
    x_train = np.sort(x[ram[:N]])
    x_test = np.sort(x[ram[N:]])

    t_train = np.sin(x_train)
    t_test = np.sin(x_test)

    return x_train, t_train, x_test, t_test

class SIN_NN(Chain):
    def __init__(self, h_units, act):
        super(SIN_NN, self).__init__()
        with self.init_scope():
            self.l1=L.Linear(1, h_units[0])
            self.l2=L.Linear(h_units[0], h_units[1])
            self.l3=L.Linear(h_units[1], 1)

            if act == "relu":
                self.act = F.relu
            elif act == "sig":
                self.act = F.sigmoid

    def __call__(self, x, t):
        x = Variable(x.astype(np.float32).reshape(x.shape[0],1))
        t = Variable(t.astype(np.float32).reshape(t.shape[0],1))

        return F.mean_squared_error(self.forward(x), t)

    def forward(self, x):
        h = self.act(self.l1(x))
        h = self.act(self.l2(h))
        h = self.l3(h)

        return h

    def predict(self, x):
        x = Variable(x.astype(np.float32).reshape(x.shape[0],1))
        y = self.forward(x)

        return y.data

# 学習（学習パラメータを設定しやすくするために関数化）
def training(N, Nte, bs, n_epoch, h_units, act):

    # データセットの取得
    x_train, t_train, x_test, t_test = get_data(N, Nte)

    # モデルセットアップ
    model = SIN_NN(h_units, act)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # loss格納配列
    tr_loss = []
    te_loss = []

    # ディレクトリを作成
    if os.path.exists("Results/{}/Pred".format(act)) == False:
        os.makedirs("Results/{}/Pred".format(act))

    # 時間を測定
    start_time = time.time()
    print("START")

    # 学習回数分のループ
    for epoch in range(1, n_epoch + 1):
        perm = np.random.permutation(N)
        sum_loss = 0
        for i in range(0, N, bs):
            x_batch = x_train[perm[i:i + bs]]
            t_batch = t_train[perm[i:i + bs]]

            model.cleargrads()
            loss = model(x_batch,t_batch)
            loss.backward()
            optimizer.update()
            sum_loss += loss.data * bs

        # 学習誤差の平均を計算
        ave_loss = sum_loss / N
        tr_loss.append(ave_loss)

        # テスト誤差
        loss = model(x_test,t_test)
        te_loss.append(loss.data)

        # 学習過程を出力
        if epoch % 100 == 1:
            print("Ep/MaxEp     tr_loss     te_loss")

        if epoch % 10 == 0:
            print("{:4}/{}  {:10.5}   {:10.5}".format(epoch, n_epoch, ave_loss, float(loss.data)))

            # 誤差をリアルタイムにグラフ表示
            plt.plot(tr_loss, label = "training")
            plt.plot(te_loss, label = "test")
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.xlabel("epoch")
            plt.ylabel("loss (MSE)")
            plt.pause(0.1)  # このコードによりリアルタイムにグラフが表示されたように見える
            plt.clf()

        if epoch % 20 == 0:
            # epoch20ごとのテスト予測結果
            plt.figure(figsize=(3, 2))
            y_test = model.predict(x_test)
            plt.plot(x_test, t_test)
            plt.plot(x_test, y_test)
            plt.legend()
            plt.grid(True)
            plt.xlim(0, 2 * np.pi)
            plt.ylim(-1.2, 1.2)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig("Results/{}/Pred/ep{}.png".format(act,epoch))
            plt.clf()
            plt.close()

    print("END")

    # 経過時間
    total_time = int(time.time() - start_time)
    print("Time : {} [s]".format(total_time))

    # 誤差のグラフ作成
    plt.figure(figsize=(4, 3))
    plt.plot(tr_loss, label = "training")
    plt.plot(te_loss, label = "test")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE)")
    plt.savefig("Results/{}/loss_history.png".format(act))
    plt.clf()
    plt.close()

    # 最終のテスト予測結果
    plt.figure(figsize=(3, 2))
    y_test = model.predict(x_test)
    plt.plot(x_test, t_test)
    plt.plot(x_test, y_test)
    plt.grid(True)
    plt.xlim(0, 2 * np.pi)
    plt.ylim(-1.2, 1.2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("Results/{}/Pred/ep{}.png".format(act,epoch))
    plt.clf()
    plt.close()

    # 最終のテスト予測結果
    plt.figure(figsize=(4, 3))
    y_test = model.predict(x_test)
    plt.plot(x_test, t_test, label = "answer")
    plt.plot(x_test, y_test, label = "prediction")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 2 * np.pi)
    plt.ylim(-1.2, 1.2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("Results/{}/Pred/ep{}_big.png".format(act,epoch))
    plt.clf()
    plt.close()

    # 学習済みモデルの保存
    serializers.save_npz("Results/{}/Model.model".format(act),model)


if __name__ == "__main__":

    # 設定
    N = 1000             # 学習データ
    Nte = 200           # テストデータ数
    bs = 10             # バッチサイズ
    n_epoch = 10#00       # 学習回数
    h_units = [10, 10]  # ユニット数 [中間層１　中間層２]
    act = "relu"        # 活性化関数(ReLU関数にしたい場合は、"relu")

    training(N, Nte, bs, n_epoch, h_units, act)
