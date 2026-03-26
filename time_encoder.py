import datetime
import torch
import torch.nn as nn
import math

def timestamp_to_features(timestamp):
    dt = datetime.datetime.utcfromtimestamp(timestamp)
    return {
        'year': dt.year,
        'month': dt.month,
        'day': dt.day,
        'hour': dt.hour,
        'minute': dt.minute,
        'second': dt.second,
        'weekday': dt.weekday()  # Monday is 0 and Sunday is 6
    }


class TimeFeatureEncoding(nn.Module):
    def __init__(self, d_model):
        super(TimeFeatureEncoding, self).__init__()
        self.hour_embed = nn.Embedding(24, d_model)
        self.minute_embed = nn.Embedding(60, d_model)
        self.second_embed = nn.Embedding(60, d_model)
        self.day_embed = nn.Embedding(31, d_model)
        self.month_embed = nn.Embedding(12, d_model)
        self.year_embed = nn.Embedding(3, d_model)  # 假设数据集包含100年的数据
        self.weekday_embed = nn.Embedding(7, d_model)

    def forward(self, time_features):
        hour_x = self.hour_embed(time_features[:, 0])
        minute_x = self.minute_embed(time_features[:, 1])
        second_x = self.second_embed(time_features[:, 2])
        day_x = self.day_embed(time_features[:, 3]-1)
        month_x = self.month_embed(time_features[:, 4]-1)
        year_x = self.year_embed(time_features[:, 5] - 2009)  # 假设年份从2000年开始
        weekday_x = self.weekday_embed(time_features[:, 6])

        # 将所有嵌入向量相加，得到时间特征的总嵌入
        return hour_x + minute_x + second_x + day_x + month_x + year_x + weekday_x



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimeEmbedding, self).__init__()
        self.time_feature_encoder = TimeFeatureEncoding(d_model)
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, time_features):
        encoded_time_features = self.time_feature_encoder(time_features)
        encoded_positions = self.pos_encoder(encoded_time_features.unsqueeze(0))
        return encoded_positions.squeeze(0)


# te=TimeFeatureEncoding(d_model=16)
# time_f=torch.tensor( [[ 14,   50,   23,    4,   12, 2010,    5]])
# out=te(time_f)
# print(out)

