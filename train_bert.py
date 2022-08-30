import os
import sys
import random
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import copy, argparse
import json, MeCab
import transformers
from transformers import BertTokenizer, BertModel
import jaconv, unicodedata, regex
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lib.util import load_data, data_augmentation, ExperimentDataset, my_collate_fn
from lib.model import BERT_POOLING, AdaCos


def main():
    # データの読み込み
    df, t_df = load_data(args)
    # 交差検証
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(skf.split(df, df["medis"])):
        logger.info("{0}-fold目の実験を開始します。".format(i))
        # 学習とテストに分割
        X_train, X_test = df.iloc[train_index], df.iloc[test_index]
        # 検証データを取得
        val_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i+1)
        for _, (train_index, val_index) in enumerate(val_skf.split(X_train, X_train["medis"])):
            X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            break
        # データを保存（評価時に使用する）
        logger.info("X_train: {0}, X_val: {1}, X_test: {2}".format(X_train.shape, X_val.shape, X_test.shape))
        X_train.to_csv("./results/{0}/X_train_{1}.csv".format(args.exp_name, i))
        X_val.to_csv("./results/{0}/X_val_{1}.csv".format(args.exp_name, i))
        X_test.to_csv("./results/{0}/X_test_{1}.csv".format(args.exp_name, i))
        # データ拡張する場合
        if args.data_aug:
            logger.info("データ拡張を実施")
            original_size = X_train.shape[0]
            X_train = data_augmentation(X_train, t_df)
            logger.info("データ拡張前のサイズ: {0}, データ拡張後のサイズ: {1}".format(original_size, X_train.shape[0]))
        else:
            logger.info("データ拡張をしない")
        # DataLoderの作成
        medis2idx = {m: i for i, m in enumerate(df["medis"].unique())}
        with open('./results/{1}/medis2idx_{0}.json'.format(i, args.exp_name), 'w') as f:
            json.dump(medis2idx, f, indent=4)
        train_dataset = ExperimentDataset(X_train["token"].to_list(), X_train["medis"].to_list(), tokenizer, medis2idx)
        val_dataset = ExperimentDataset(X_val["token"].to_list(), X_val["medis"].to_list(), tokenizer, medis2idx)
        test_dataset = ExperimentDataset(X_test["token"].to_list(), X_test["medis"].to_list(), tokenizer, medis2idx)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
        # モデルを訓練
        train_val_loop(train_loader, val_loader, medis2idx, i)


def train_val_loop(train_loader, val_loader, tag2idx, fold):
    # 訓練
    best_val_loss =  1e7
    # モデルを定義
    model = BERT_POOLING(args, tag2idx).to(device)
    # 最適化関数
    optimizer = optim.AdamW([
                            {'params': model.bert_model.parameters(), 'lr': 3e-5, 'weight_decay': 0.01},
                            {'params': model.linear.parameters(), 'lr': 1e-3, 'weight_decay': 0.01}],
                            )
    warmup_steps = int(args.max_epoch * len(train_loader) * 0.1 / args.batch_size)
    # warmup_steps = 0
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                                             num_warmup_steps=warmup_steps, 
                                                             num_training_steps=len(train_loader)*args.max_epoch)
    # BERTの全レイヤーの勾配を更新
    for _, param in model.named_parameters():
        param.requires_grad = True
    # マルチGPU
    model = torch.nn.DataParallel(model)
    # 損失関数を定義
    loss = nn.CrossEntropyLoss()

    # 以下、訓練ループ
    for epoch in range(args.max_epoch):
        # 訓練モード
        model.train()
        # 損失　
        train_running_loss = 0
        # バッチ処理
        pbar_train = tqdm(train_loader)
        for batch in pbar_train:
            # 勾配を初期化
            optimizer.zero_grad()
            # to tensor
            train_vecs = pad_sequence(batch[0], padding_value=0, batch_first=True).to(device)
            train_labels = torch.tensor(batch[1]).to(device)
            # 予測
            logits, _ = model(train_vecs)
            # 損失の計算
            model_loss = loss(logits, train_labels)
            # 誤差伝搬
            model_loss.backward()
            # 勾配を更新
            optimizer.step()
            scheduler.step()
            # 合計の損失
            train_running_loss += model_loss.item()
        # https://aidiary.hatenablog.com/entry/20180204/1517705138
        train_running_loss = train_running_loss / len(train_loader)
        logger.info("訓練")
        logger.info("{0}エポック目の損失値: {1}\n".format(epoch, train_running_loss))

        # 検証
        with torch.inference_mode():
            # 検証モード
            model.eval()
            val_running_loss = 0
            pbar_val = tqdm(val_loader)
            for batch in pbar_val:
                # to tensor
                val_vecs = pad_sequence(batch[0], padding_value=0, batch_first=True).to(device)
                val_labels = torch.tensor(batch[1]).to(device)
                # 予測
                val_logits, _ = model(val_vecs)
                # 損失の計算
                model_loss = loss(val_logits, val_labels)
                # 合計の損失
                val_running_loss += model_loss.item()
        val_running_loss = val_running_loss / len(val_loader)
        # 保存
        logger.info("検証")
        logger.info("{0}エポック目の損失値: {1:.4}".format(epoch, val_running_loss))

        # Early Stopping
        if val_running_loss < best_val_loss:
            logger.info("{0}エポック目で損失を更新\n".format(epoch))
            torch.save(model.module.state_dict(), './models/{1}/model_{0}.pt'.format(fold, args.exp_name))
            best_val_loss = val_running_loss
        else:
            logger.info("{0}エポック目は現状維持\n".format(epoch))


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('use cuda device')
        seed=777
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        print('use cpu')
        device = torch.device('cpu')
        torch.manual_seed(999)
        np.random.seed(999)
    # 引数
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str, default='../BERT/UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K')
    parser.add_argument('--data_path', type=str, default="./data/train/train.csv")
    parser.add_argument('--tdct_path', type=str, default="./data/train/t_df.csv")
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--exp_name', type=str, default="bert_cls")
    parser.add_argument('--label', type=str, default="medis")
    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--icd', action='store_true')
    args = parser.parse_args()
    
    # フォルダ作成
    # for SAMPLE_DIR in ["./logs/{0}".format(args.exp_name), "./models/{0}".format(args.exp_name), "./results/{0}".format(args.exp_name)]:
    for SAMPLE_DIR in ["./models/{0}".format(args.exp_name), "./results/{0}".format(args.exp_name)]:
        if not os.path.exists(SAMPLE_DIR):
            os.makedirs(SAMPLE_DIR)

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s- %(lineno)d - %(funcName)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger('LoggingTest')
    logger.setLevel(10)
    sh = logging.StreamHandler()
    logger.addHandler(sh)
    fh = logging.FileHandler('./logs/{0}.log'.format(args.exp_name), "w")
    logger.addHandler(fh)
    formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    tokenizer = BertTokenizer(Path(args.bert_path) / "vocab.txt", do_lower_case=False, do_basic_tokenize=False)

    main()