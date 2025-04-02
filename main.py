from data_loader import collect_labels, load_and_convert_all
from gnn_model import SimpleGNN
from train import train
from random import shuffle
from data_loader import collect_labels, load_and_convert_all, get_common_feature_set
# Veriyi hazırla
csv_dir = "Data"
# Etiket ve ortak öznitelikleri topla
app_map, cat_map = collect_labels(csv_dir)
common_features = get_common_feature_set(csv_dir)
# Veriyi dönüştür
graph_list = load_and_convert_all(csv_dir, app_map, cat_map, common_features)
# Modeli başlat
in_channels = graph_list[0].x.shape[1]
model = SimpleGNN(in_channels=in_channels, hidden_channels=512, out_channels=len(app_map))
shuffle(graph_list)
# %70 train, %15 val
train_size = int(0.7 * len(graph_list))
val_size = int(0.15 * len(graph_list))
train_data = graph_list[:train_size]
val_data = graph_list[train_size:train_size + val_size]
# Eğitimi başlat
train(model, train_data, val_data, epochs=30)

