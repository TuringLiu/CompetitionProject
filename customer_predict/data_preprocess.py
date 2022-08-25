import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

dataset_path = './dataset/'
train_data_path = dataset_path + 'train.csv'
result_data_path = dataset_path + 'test.csv'

class_feature = ['地理区域', '是否双频', '是否翻新机', '手机网络功能', '婚姻状况',
       '信息库匹配', '预计收入', '信用卡指示器', '新手机用户']


'sample数据，然后划分训练集，测试集'
def get_dataset(data_path=train_data_path, sample=False, sample_num=500, test_size=0.25, random_state=0):
    data_df = pd.read_csv(data_path)
    # 去除客户ID
    data_df = data_df.drop('客户ID', axis=1)
    # 为了保证模型验证的稳定性，先使用固定的sample，后面测试效果时，可采用random sample模式
    if sample:
        data_df = data_df[: sample_num]
    x = data_df.iloc[:, :-1]
    y = data_df['是否流失']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, x_test, y_train, y_test

'划分连续特征，离散特征'
# 返回类别特征的维度，未使用
def get_class_dimension(all_df, class_feature=class_feature):
    class_dim = defaultdict(int)
    for c_f in class_feature:
        class_dim[c_f] = len(all_df[c_f].unique())
    return class_dim

# 使用全量数据集（包括test.csv）得到onehot 编码器
def get_onehot_encoder(class_feature=class_feature):
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(result_data_path)
    train_df = train_df.iloc[:, :-1]
    all_df = pd.concat([train_df, test_df], axis=0)

    # 对类别特征进行one-hot操作
    class_label_dict = defaultdict()
    class_onehot_dict = defaultdict()
    # 得到每一个类别的onehot encoder
    for c_f in class_feature:
        feat = all_df[c_f]
        label_f = LabelEncoder().fit(feat)
        onehot_f = OneHotEncoder().fit(label_f.transform(feat).reshape(-1, 1))
        class_label_dict[c_f] = label_f
        class_onehot_dict[c_f] = onehot_f
    return class_label_dict, class_onehot_dict

'对类别特征做one-hot操作, 返回onehot字典'
def split_feature_type(x, class_label_dict, class_onehot_dict, class_feature=class_feature):
    onehot_dict = defaultdict()
    for cf in class_feature:
        data = x[cf]
        label_f = class_label_dict[cf]
        onehot_f = class_onehot_dict[cf]

        data_label = label_f.transform(data)
        data_onehot = onehot_f.transform(data_label.reshape(-1, 1)).toarray()
        onehot_dict[cf] = data_onehot
    return onehot_dict


class DeepDataSet(Dataset):
    def __init__(self, x: 'DataFrame', y: 'Series', class_feature: list, class_label_dict, class_onehot_dict,):
        self.x = x
        self.y = np.array(y, dtype=np.float)
        self.class_feature = class_feature
        self.value_feature = [k for k in x_train.columns.values.tolist() if k not in class_feature]
        self.class_dict = split_feature_type(x, class_label_dict, class_onehot_dict)
        self.value_np = np.array(x[self.value_feature], dtype=np.float)
        print(self.value_np.shape)


    def __getitem__(self, index):
        '''
        :param index:
        :return:
        x_class_dict: dict['str'] = np.ndarray
        value_feature: DataFrame
        label: Series
        '''
        label = self.y[index]
        x_class_dict = defaultdict()
        for cf in self.class_feature:
            data = self.class_dict[cf]
            x_class_dict[cf] = data[index, :]
        print(index, type(label))
        print(type(x_class_dict), type(self.value_np[index, :]), type(label))
        return x_class_dict, self.value_np[index, :], label

    def __len__(self):
        return len(self.y)


x_train, x_test, y_train, y_test = get_dataset(sample=True)
# print(x_train[1])
class_label_dict, class_onehot_dict = get_onehot_encoder()
onehot_dict = split_feature_type(x_train, class_label_dict, class_onehot_dict)

train_dataset = DeepDataSet(x_train, y_train, class_feature, class_label_dict, class_onehot_dict)
train_loader = DataLoader(train_dataset, batch_size=1)



for x_class_dict, value_df, label in tqdm(train_loader):
    print(x_class_dict, value_df, label)
    input()