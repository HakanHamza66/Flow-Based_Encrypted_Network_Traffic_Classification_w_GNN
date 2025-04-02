import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
from itertools import product

# Ortak etiket eşlemelerini çıkartan fonksiyon
def collect_labels(csv_folder):
    app_names = set()
    categories = set()
    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            path = os.path.join(csv_folder, file)
            df = pd.read_csv(path)
            app_names.update(df['appName'].unique())
            categories.update(df['category'].unique())

    app_map = {name: idx for idx, name in enumerate(sorted(app_names))}
    cat_map = {cat: idx for idx, cat in enumerate(sorted(categories))}
    return app_map, cat_map

# Ortak tüm feature setini çıkar
def get_common_feature_set(csv_folder):
    all_features = set()
    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(csv_folder, file))
            numeric = df.drop(columns=['appName', 'category'], errors='ignore')
            all_features.update(numeric.columns)
    return sorted(list(all_features))

# Veriyi yükleyip GNN için Data nesnelerine dönüştüren fonksiyon
def load_and_convert_all(csv_folder, app_map, cat_map, common_features):
    graph_list = []
    scaler = MinMaxScaler()

    # Tüm verileri toplayıp normalize etmek için birleştir
    all_data = []
    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(csv_folder, file))
            numeric = df.reindex(columns=common_features).fillna(0).astype('float32')
            all_data.append(numeric)
    full_data = pd.concat(all_data)
    scaler.fit(full_data)

    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            path = os.path.join(csv_folder, file)
            df = pd.read_csv(path)

            numeric = df.reindex(columns=common_features).fillna(0).astype('float32')
            numeric[:] = scaler.transform(numeric)

            for i, row in df.iterrows():
                x_values = numeric.iloc[i].values.astype('float32')

                num_features = len(x_values)
                if num_features % 15 != 0:
                    padding = 15 - (num_features % 15)
                    x_values = np.concatenate([x_values, np.zeros(padding, dtype='float32')])

                num_nodes = len(x_values) // 15
                x = torch.tensor(x_values, dtype=torch.float).view(num_nodes, 15)

                # Tam bağlantılı edge_index oluştur (her node her node'a bağlı)
                edge_index = torch.tensor(
                    list(product(range(num_nodes), repeat=2)), dtype=torch.long
                ).t().contiguous()

                y = torch.tensor([app_map[row['appName']]], dtype=torch.long)
                data = Data(x=x, edge_index=edge_index, y=y)
                data.category = torch.tensor([cat_map[row['category']]], dtype=torch.long)
                graph_list.append(data)
    return graph_list

