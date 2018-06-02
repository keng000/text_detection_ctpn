# Description
CTPN(Connectionist Text Proposal Network)とは、画像内から文字列を検出するニューラルネットワークのモデル名である．  

## Reference
* Zhi Tian: Detecting Text in Natural Image with Connectionist Text Proposal Network  
https://arxiv.org/abs/1609.03605  

* tianzhi0549/CTPN  
https://github.com/tianzhi0549/CTPN


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

# Setup
## Requirements
* python >= 3.5  
* CUDA == 8.0  
* cuDNN == 6.x 

## GPUを使う場合

1. プロジェクトクローン

    ```
    git clone https://github.com/keng000/text-detection-ctpn.git text_detection_ctpn
    ```

1. 学習済みモデルを取得．パスをconfigに記載する．

	学習済みモデルのダウンロードリンクは下記．  
	https://drive.google.com/open?id=18EMw2lyXekqbDYxhf-ewrUsShAqxls_4
    
	checkpoints/ へのパスを記す．

    ```
    vim text_detection_ctpn/ctpn/text.yml
    TEST:
      checkpoints_path: checkpoints/ # -> 更新
     
    ```

1. requirements.txtインストール

    ```
    pip install -r text_detection_ctpn/requirements.txt
    ```

1. Cythonスクリプトのコンパイル
    
    ```
    cd text_detection_ctpn/lib/utils
    sh make.sh gpu
    ```

1. site-packages/ から text_detection_ctpn/へシムリンクを貼る
	
	```
	cd text_detection_ctpn
	ln -si `pwd` `python -c 'import os.path as d; import pip; print(d.dirname(d.dirname(pip.__file__)))'`/$(basename `pwd`)
	```

## GPUを使わない場合


1. プロジェクトクローン

    ```
    git clone https://github.com/keng000/text-detection-ctpn.git text_detection_ctpn
    ```

1. 学習済みモデルを取得．パスをconfigに記載する．

	学習済みモデルのダウンロードリンクは下記．  
	https://drive.google.com/open?id=18EMw2lyXekqbDYxhf-ewrUsShAqxls_4
    
	checkpoints/ へのパスを記す．

    ```
    vim text_detection_ctpn/ctpn/text.yml
    TEST:
      checkpoints_path: checkpoints/ # -> 更新
     
    ```

1. GPUフラグをFalseにする．

    ```
    vim text_detection_ctpn/ctpn/text.yml
    USE_GPU_NMS: True # -> False
    ```

    ```
    vim text_detection_ctpn/lib/fast_rcnn/config.py
    __C.USE_GPU_NMS = True # -> False
    ```

1. requirements.txtインストール
	
	tensorflow-gpu を tensorflowに変える．

    ```
	vim requirements.txt
	tensorflow-gpu==1.3.0 -> tensorflow==1.3.0
	```

	```
    pip install -r text_detection_ctpn/requirements.txt
    ```

1. Cythonスクリプトのコンパイル
    
    ```
    cd text_detection_ctpn/lib/utils
    sh make.sh cpu
    ```

1. site-packages/ から text_detection_ctpn/へシムリンクを貼る
	
	```
	cd text_detection_ctpn
	ln -si `pwd` `python -c 'import os.path as d; import pip; print(d.dirname(d.dirname(pip.__file__)))'`/$(basename `pwd`)
	```

## 動作検証

サンプルプログラムによる動作検証
```
python ctpn/demo.py
```

期待する出力
```
Tensor("Placeholder:0", shape=(?, ?, ?, 3), dtype=float32)
Tensor("conv5_3/conv5_3:0", shape=(?, ?, ?, 512), dtype=float32)
Tensor("rpn_conv/3x3/rpn_conv/3x3:0", shape=(?, ?, ?, 512), dtype=float32)
Tensor("lstm_o/Reshape_2:0", shape=(?, ?, ?, 512), dtype=float32)
Tensor("lstm_o/Reshape_2:0", shape=(?, ?, ?, 512), dtype=float32)
Tensor("rpn_cls_score/Reshape_1:0", shape=(?, ?, ?, 20), dtype=float32)
Tensor("rpn_cls_prob:0", shape=(?, ?, ?, ?), dtype=float32)
Tensor("Reshape_2:0", shape=(?, ?, ?, 20), dtype=float32)
Tensor("rpn_bbox_pred/Reshape_1:0", shape=(?, ?, ?, 40), dtype=float32)
Tensor("Placeholder_1:0", shape=(?, 3), dtype=float32)
Loading network VGGnet_test...  Restoring from /path/to/text_detection_ctpn/checkpoints... done
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo for /path/to/lib/python3.6/site-packages/text_detection_ctpn/data/demo/010.png
Detection took 4.965s for 8 object proposals       
```

# TODO
- [ ] setpy.pyによるモジュールのセットアップ化


