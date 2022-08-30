# 患者状態表現の病名交換コードへのマッピング

本リポジトリは第42回医療情報学連合大会で発表した「患者状態表現の病名交換コードへのマッピング」の実験コードです。詳細抄録に掲載されているデータ拡張を用いた実験の再現にはティ辞書企画さまより販売されている[シソーラス辞書](https://www.tdic.co.jp/)が必要となります。

## 自前準備 

1. [ICD10対応標準病名マスター](http://www2.medis.or.jp/stdcd/byomei/index.html)をダウンロードし、./data/medisにindex504.txtとnmain504.csvを置く。
2. [シソーラス辞書](https://www.tdic.co.jp/)を使用する場合、tdic_tenkai202111_UTF8.csvを./data/tに置く。
3. [万病辞書](http://sociocom.jp/~data/2018-manbyo/index.html)をダウンロードし、./data/manbyoにMANBYO_20210602.csvを置く。
4. python preprocess.py --data_augを実行する（シソーラス辞書がない場合は--data_augをつけない）
5. [UTH-BERT](https://ai-health.m.u-tokyo.ac.jp/home/research/uth-bert)をダウンロードし、./BERTに置く。

※ フォルダがない場合は以下で作成してください。
```
mkdir ./data ./BERT ./data/t ./data/medis ./data/manbyo 
```

## 学習

### 機械学習モデルの訓練
```
bash train_medsi.sh
```

## 評価

### UNKを考慮しない場合

```
bash eval_medis_wo_unk.sh
```

### UNKを考慮する場合

```
bash eval_medis_w_unk.sh
```

### ベースライン
ベースラインを実行する前に必ず機械学習モデルの訓練を実行してください。

```
# データ拡張なし
python baseline.py --data_path ./results/bert_cls_medis
# データ拡張あり
python baseline.py --data_path ./results/bert_cls_medis_aug
```

## 参考

- simstring available from https://github.com/nullnull/simstring.

## 引用
本リポジトリを参照する場合は以下の文献を引用してください。
```
```
