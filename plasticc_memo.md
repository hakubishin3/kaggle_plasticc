# 1st solution
https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75033  
- 5foldのlightgbm（single）
- テストデータの特性の合わせるよう、訓練データをaugment
- 特徴抽出のためにGPを使ってlight-curve fitting（Matern Kernel）
- データ点数がpoorなobjectの場合、GPの結果が良くない可能性が高い。objectのfluxが最大となる時点周辺のデータ点数を数えて、GPによるfittingの指標として使った。

# 2nd solution
https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75059  
- max_depth=2のlightgbmモデルとのensembleが良かった。
- hostgal_speczを予測するモデルを作成、そのoof predを利用した。
- data augmentationを行った。NNモデルではスコア向上したが、lightgbmでは役に立たなかった。

# 3rd solution
https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75116  
https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75131  
https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75222  

- catboost
  - 一部のパラメータのみ調整（colsampleとか）
  - catboost > xgboost > lightgbmの順で精度が高かった。各パッケージのパラメータのデフォルト値による違い？
  - 特徴数は約2000個。計算時間の関係で数減らした。多少overfitしていても、後のensembleでの伸びしろが大きくなる。
  - curve angle - np.arctan(flux_diff/mjd_diff) が効いた。

- lightgbm
  - sncosmoパッケージを使ったtemplate fitting。https://www.kaggle.com/nyanpn/salt-2-feature-part-of-3rd-place-solution
  - 計算がsegmentation errorを吐く場合があったので、object_id%30でデータを分割し、30インスタンスで並列計算。
  - 様々なtemplateを使用して（salt-2, salt-2-extended, sako, ...）、そのうち8つを最終的に採用。

- CNN
  - https://www.kaggle.com/yuval6967/3rd-place-cnn
  - 1d-cnnを用いたネットワーク構成。次の資料の構成に似ている。https://arxiv.org/pdf/1611.06455.pdf
  - cnnの最終層出力とmeta-data用のMLP出力をconcat。
  - 各チャンネルのlight-curveを線形補間
  - 補間した値の信頼度を追加（補間時点と最も近いデータ点との距離を信頼度として与える）
  - conv1dでstandard scalerをするとスパースが崩れる。stdで割るだけ。
  - standard scalerよりもrank gaussがよかった（？）

- augmentation
  - 最大30%まで、データ点をランダムに除外（線形補間前）
  - flux_errに応じてfluxをランダムに変化（線形補間前）
  - cyclic shift。後ろ方向にずらしたデータは前方にくっつける。（線形補間後）
  - skew。全チャンネルのfluxに対して(1+k)倍掛ける。kは乱数（線形補間後）

- class99
  - (1-p)にかかる最適なべき指数をLBで見つけた。

- pseudo-labeling
    - class90だけをtestから引っ張ることでスコア向上。
    - adversarial pseudo labeling（adversarial validationでtrainと判定されたものかつ、クラス分類の予測値が高いものを使う）がNNに効いた。
    - LGBM -> LGBMは効かない。LGBM->NNは効いた。でも逆はいまいち。

# 4th solution
https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75011  
- autoencoderモデルを作成したが、特徴量として役に立たなかった。

# 21st solution
https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75156  
- GPyOptを使って最適なsample weightを探す。classごとに、データ数に合わせてweightの探索範囲を決めておく。

# other
- boosting全般
- 5foldでoof作って、全部で学習してテストの予測作る。
- この際に、epoch数は5foldなら(1.5*5/4)倍がいい。(1.3派の人もいる)
- ただ倍率は収束するepoch数によって1000とかならいいと思うけど100とかだとやりすぎ。
- epoch数が増える特徴がいい特徴(学習率は同じ)
- 特徴の数は多いほどアンサンブルで上がる、減らす理由は時間がかかる以外にない。
- lambda, alphaも小さい方がアンサンブルにはいい

