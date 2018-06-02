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

