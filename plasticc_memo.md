# 1st solution



# 3rd solution
https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75116  
https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75131  
https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75222  

- catboost
  - 一部のパラメータのみ調整（colsampleとか）
  - catboost > xgboost > lightgbmの順で精度が高かった。各パッケージのパラメータのデフォルト値による違い？
  - 特徴数は約2000個。計算時間の関係で数減らした。多少overfitしていても、後のensembleでの伸びしろが大きくなる。
  - curve angle - np.arctan(flux_diff/mjd_diff) が効いた。

- class99
  - (1-p)にかかる最適なべき指数をLBで見つけた。

- pseudo-labeling
    - class90だけをtestから引っ張ることでスコア向上
    - adversarial pseudo labeling（adversarial validationでtrainと判定されたものかつ、クラス分類の予測値が高いものを使う）がNNに効いた。
    - LGBM -> LGBMは効かない。LGBM->NNは効いた。でも逆はいまいち。


# other

・boosting全般
5foldでoof作って、全部で学習してテストの予測作る。
この際に、epoch数は5foldなら(1.5*5/4)倍がいい。(1.3派の人もいる)
ただ倍率は収束するepoch数によって1000とかならいいと思うけど100とかだとやりすぎ。
epoch数が増える特徴がいい特徴(学習率は同じ)
特徴の数は多いほどアンサンブルで上がる、減らす理由は時間がかかる以外にない。
lambda, alphaも小さい方がアンサンブルにはいい

・NN
Conv1dでStandard Scalerをするとスパースが崩れるからダメ。stdで割るだけ
Standard ScalerよりRank Gaussの方がよかったのは謎
自信度->補間時の近い点との距離を自信度として与えるとスコア上がる

・sample weightの調整
これが効くということは、LGBM完璧じゃない
クラスごとにデータ数に合わせてweightの探索範囲は変える

・SMOTE
fluxに使うと効く。
downもupも同時にやった。

・post processing
trainのデータ数でpred*weight
(pred*weight1)**weight2はoverfitした
