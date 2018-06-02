# Description
CTPN(Connectionist Text Proposal Network)とは、画像内から文字列を検出するニューラルネットワークのモデル名である．  

## Reference
* Zhi Tian: Detecting Text in Natural Image with Connectionist Text Proposal Network  
https://arxiv.org/abs/1609.03605  

* tianzhi0549/CTPN  
https://github.com/tianzhi0549/CTPN

## Requirements
* python >= 3.5  
* CUDA == 8.0  
* cuDNN == 6.x 

# Setup
[セットアップ手順](MODULE_SETUP.md)

# Usage 
## Functions

### ctpn_interface.ctpn(sess, net, in_img)

このモジュールのメインとなる関数．  
入力した画像を元に、その画像上の文字列と思しき領域の座標をリストで返す．

入力 

| Args | Type | Remarks |
| :---: | :---: | :--- |
| sess | tensorflow.Session | Tensorflowセッション. 下記tf_utils.create_tf_session()により生成． |
| net | network.Network | Tensorflow計算グラフ. 下記tf_utils.load_trained_model(sess)により生成． |
| in_img | numpy.ndarray | ３チャンネル入力画像． |

出力

| Type | Remarks |
| :---: | :--- |
| list of tuple | 1つの要素が座標を表すタプル (左上x, 左上y, 右下x, 右下y) であるリスト |


### tf_utils.create_tf_session()

Tensorflowのセッションを生成する関数．

 出力

| Type | Remarks |
| :---: | :--- |
| sess | tensorflow.Session | Tensorflowセッション. |

### tf_utils.load_trained_model(sess)

Tensorflowのセッション上に学習済み計算グラフを展開し、計算グラフを返す関数．

入力 

| Args | Type | Remarks |
| :---: | :---: | :--- |
| sess | tensorflow.Session | Tensorflowセッション. 下記tf_utils.create_tf_session()により生成． |

出力

| Type | Remarks |
| :---: | :--- |
| network.Network | Tensorflow計算グラフ. |


# TODO
- [ ] setpy.pyによるモジュールのセットアップ化


