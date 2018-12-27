・catboost
パラメータ2つしか変えてない
colsampleと？？
catboost>xgb>lgbmだったのは単にパラメータ説
float64指定はやばい
カテゴリー使わなきゃ普通にはやい

・xgb
exact指定してもめちゃ早くなった
精度もいい

・LGBM
sub sample指定するだけじゃ変わらない
float64指定はやばい

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

・特徴
curve angle良い

・adversarial pseudo labeling
adversarial validationでtrainと判定されたものかつ、クラス分類の予測値が高いもの(0.99以上とか)を使う。NNに効いた。
pseudo labelingはLGBM->LGBMでは効かないのは、同じモデルだから。LGBM->NNは効いた。でも逆はいまいち。

・sample weightの調整
これが効くということは、LGBM完璧じゃない
クラスごとにデータ数に合わせてweightの探索範囲は変える

・SMOTE
fluxに使うと効く。
downもupも同時にやった。

・post processing
trainのデータ数でpred*weight
(pred*weight1)**weight2はoverfitした

・その他
logしてaggregationはスケール変わる
jupyter or .py
