# Python 環境の作成
python 3.6.9にて実施

```bash
$ python -m venv env
$ source env/bin/activate
$ pip install -U pip
```

# Bottom-up-attentionのインストール
Bottom-up-attentionのclone

```bash
git clone git@github.com:peteanderson80/bottom-up-attention.git
```

Bottome-up-attentionのセットアップ

```bash
$ cd bottom-up-attention
```

# データセットの作成

データセット作成用作業ディレクトリの作成

```bash
$ cd ../data
$ mkdir vg
``

Visual Genomeのデータ( https://visualgenome.org/api/v0/api_home.html )のダウンロード

```bash
$ cd vg
$ wget https://visualgenome.org/static/data/dataset/objects.json.zip
$ wget https://visualgenome.org/static/data/dataset/relationships.json.zip
$ wget https://visualgenome.org/static/data/dataset/object_alias.txt
$ wget https://visualgenome.org/static/data/dataset/relationship_alias.txt
$ wget https://visualgenome.org/static/data/dataset/object_synsets.json.zip
$ wget https://visualgenome.org/static/data/dataset/attribute_synsets.json.zip
$ wget https://visualgenome.org/static/data/dataset/relationship_synsets.json.zip
$ wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
$ wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
$ wget https://visualgenome.org/static/data/dataset/image_data.json.zip
$ wget https://visualgenome.org/static/data/dataset/region_descriptions.json.zip
$ wget https://visualgenome.org/static/data/dataset/question_answers.json.zip
$ wget https://visualgenome.org/static/data/dataset/attributes.json.zip
$ wget https://visualgenome.org/static/data/dataset/synsets.json.zip
$ wget https://visualgenome.org/static/data/dataset/region_graphs.json.zip
$ wget https://visualgenome.org/static/data/dataset/scene_graphs.json.zip
$ wget https://visualgenome.org/static/data/dataset/qa_to_region_mapping.json.zip
$ for i in *.zip;do unzip $i;done

```

データセットの前処理の実行

```bash
$ cd ../../ 
$ python data/genome/setup_vg.py
```
# faster-rcnn.pytorchのインストール

```bash
$ cd ..
$ git clone git@github.com:jwyang/faster-rcnn.pytorch.git
$ cd faster-rcnn.pytorch
$ git checkout -b origin/visual-genome
```

依存ライブラリのインストール

```bash
$ pip install scipy==1.1 torch==1.0.0 torchvision==0.2.2 pycocotools
$ pip install -r requirements.txt
```

ライブラリのビルド

```bash
$ cd lib
$ python setup.py build develop
$ cd ..
```

Visual Genomeデータセットへの参照作成

```bash
$ mkdir data
$ cd data
$ ln -s  ../../bottom-up-attention/data/genome .
$ ln -s ../../bottom-up-attention/data/vg .
$ cd ..
```

ベースモデルの設置

```bash
$ mkdir data/pretrained_model
```

作成した `data/pretrained_model` ディレクトリに `resnet101_caffe.pth` を設置する。

# 学習の実行

下記のコマンドで学習を実行する。GPU一枚のときは `--mGPUs` オプションは省略する。
```bash
$ python trainval_net.py --dataset vg --net res101 --bs 16 --nw 4 --lr 1e-3 --lr_decay_step 5 --cuda --mGPUs
```
完了すると `models/res101/vg/` 以下に `faster_cnn_1_20_12145.pth` などの名前でモデルが保存される。なお `faster_cnn_(session)_(epoch)_(checkpoint).pth` という命名規則になっている。

# 推論の実行

学習したモデルを参照するようにする。
```bash
$ cd data/pretrained_model
$ ln -s ../../models/res101/vg/faster_rcnn_1_20_12145.pth .
$ cd ../../
```
推論を実行する
```
$ python demo.py --net res101 --checksession 1  --checkepoch 20 --checkpoint 48915 --dataset vg --cfg cfgs/res101.yml  --cuda --load_dir data/pretrained_model/ --image_dir data/objects
```

`--checksession` `--chechepoch` `--checkpoint` にて読み込むモデルを指定するので学習したモデルに該当する数字を指定する必要がある。
`--image_dir` 以下に存在する画像ファイルに対して実行し、同じディレクトリに `元のファイル名_det.jpg` というファイル名で結果ファイルが作成される。

# 別の学習済みモデルを用いる場合

 `faster-rcnn.pytorch/data/pretrained_model` 以下に `faster_rcnn_1_20_48915.pth` をコピーする。推論実行時に `--checkpoint` を `48915` に指定する。 
