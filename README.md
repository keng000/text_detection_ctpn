# CTPNモジュールとは
CTPT(Connectionist Text Proposal Network)とは、画像内から文字列を検出するニューラルネットワークのモデル名である。  
参考: https://arxiv.org/abs/1609.03605
参考: https://github.com/tianzhi0549/CTPN

# セットアップ
## GPUを使う場合

1. プロジェクトクローン

    ```
    git clone https://github.com/keng000/text-detection-ctpn.git text_detection_ctpn
    ```

1. 学習済みモデルのパスをconfigに記載
    checkpoints/ へのパスを記す。
    ```
    vim text_detection_ctpn/ctpn/text.yml
    TEST:
      checkpoints_path: checkpoints/ # -> 更新
     
    ```

1. requirements.txtインストール
ただし、kenichiにはopencvが既にインストールされているため、 opencv-pythonは消す。
 
    ```
    pip install -r text_detection_ctpn/requirements.txt
    ```

1. Cythonスクリプトのコンパイル
    
    ```
    cd text_detection_ctpn/lib/utils
    sh make.sh gpu
    ```

## GPUを使わない場合


1. プロジェクトクローン

    ```
    git clone https://github.com/keng000/text-detection-ctpn.git text_detection_ctpn
    ```

1. GPUフラグをFalseにする。

    ```
    vim text_detection_ctpn/ctpn/text.yml
    USE_GPU_NMS: True # -> False
    ```

    ```
    vim text_detection_ctpn/lib/fast_rcnn/config.py
    __C.USE_GPU_NMS = True # -> False
    ```

1. 学習済みモデルのパスをconfigに記載
    checkpoints/ へのパスを記す。
    ```
    vim text_detection_ctpn/ctpn/text.yml
    TEST:
      checkpoints_path: checkpoints/ # -> 更新
     
    ```

1. requirements.txtインストール
ただし、kenichiにはopencvが既にインストールされているため、 opencv-pythonは消す。
またtensorflow-gpu を tensorflowに変える。

    ```
    pip install -r text_detection_ctpn/requirements.txt
    ```

1. Cythonスクリプトのコンパイル
    
    ```
    cd text_detection_ctpn/lib/utils
    sh make.sh cpu
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
Loading network VGGnet_test...  Restoring from /home/kengo/workspace/python3X/github_library/text_detection_ctpn/checkpoints... done
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo for /home/kengo/anaconda3/envs/klavpy3Env/lib/python3.6/site-packages/text_detection_ctpn/data/demo/010.png
Detection took 4.965s for 8 object proposals       
```

