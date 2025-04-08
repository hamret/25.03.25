import torch
import pandas as pd
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import logging
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("Using device: ", device)

# 1. 학습시 경고 메세지 제거
logging.set_verbosity_error()

# 2. 데이터 확인
path = "C:\dev\mobiebert project\hct.csv"
try:
    df = pd.read_csv(path, encoding="utf-8")
except UnicodeDecodeError:
    try:
        df = pd.read_csv(path, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", errors='ignore')
        print("경고: 파일 인코딩을 latin-1으로 처리했으며, 데이터 손실이 발생할 수 있습니다.")
except pd.errors.ParserError as e:
    print(f"CSV 파싱 에러: {e}")
    raise

# 필요한 열 확인 및 추출
if 'comment_text' not in df.columns or \
   'severe_toxic' not in df.columns or \
   'threat' not in df.columns or \
   'toxic' not in df.columns or \
   'toxicity' not in df.columns:
    print("오류: 필요한 열 ('comment_text', 'severe_toxic', 'threat', 'toxic', 'toxicity')이 DataFrame에 없습니다.")
    exit()

data_X = list(df['comment_text'].values)
true_labels = df[['severe_toxic', 'threat', 'toxic', 'toxicity']].values
print(data_X[:5])
# test.py