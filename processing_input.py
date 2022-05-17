import torch
from transformers import AutoModel, AutoTokenizer
from underthesea import word_tokenize
import re
import pickle
import numpy as np

phobert = AutoModel.from_pretrained("vinai/phobert-base", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-token-base", local_files_only=True)
model = pickle.load(open('Model_SVM.pkl', 'rb'))
path_file_acr = "./BuildModel/TuDienVietTat/acronym_vi.txt"
acronyms_list = []


def tokenizer_vn(row):
    return word_tokenize(row, format="text")


with open(path_file_acr, 'r', encoding="utf8") as f:
    for line in f.readlines():
        line = re.sub('[\n]', '', line)
        acronyms_list.append(line.split('\t'))


def replace_acronyms(self, acronyms):
    list_text = self.split(" ")
    for i in range(len(list_text)):
        for j in range(len(acronyms)):
            if (list_text[i] == acronyms[j][0]):
                list_text[i] = acronyms[j][1]
    self = " ".join(list_text)
    return self


# Hàm chuẩn hoá câu
def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    row = re.sub('[\n\/]', '', row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", "").replace(".", "") \
        .replace(";", "").replace("“", "") \
        .replace(":", "").replace("”", "") \
        .replace('"', "").replace("'", "") \
        .replace("!", "").replace("?", "") \
        .replace("-", "").replace("?", "") \
        .replace('*\r?\n*', '')

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    row = emoji_pattern.sub(r'', row)
    row = row.strip().lower()
    return row


def feature_text(text):
    max_len = 20
    text = standardize_data(text)
    text = replace_acronyms(text, acronyms=acronyms_list)
    sentence = tokenizer_vn(text)
    tokenized = tokenizer.encode(sentence)
    padded = np.array([tokenized + [0] * (max_len - len(tokenized))])
    train_mask = np.where(padded == 0, 0, 1)
    train_text = torch.tensor(padded).to(torch.long)
    train_mask = torch.tensor(train_mask)
    with torch.no_grad():
        features = phobert(train_text, train_mask)
    return features[0][:, 0, :].numpy()

def pre_comment(text):
    features = feature_text(text)
    sc = model.predict(features)
    if(sc[0] == 0):
        return "Tích cực"
    else:
        return "Tiêu cực"


if __name__ == '__main__':
    text = "Thời gian ship lâu, sản phẩm đóng gói kém"
    pre_comment(text)