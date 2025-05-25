import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
val_f1_scores = []
def should_stop_early(metrics, patience):
    best = max(metrics)
    best_index = metrics.index(best)
    if len(metrics) - best_index - 1 >= patience:
        return True
    return False
def train(model, train_data, val_data, epochs=30, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc, prec, rec, f1 = evaluate(model, val_loader)
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f}")
        val_f1_scores.append(f1)

        if should_stop_early(val_f1_scores, patience=7):
            print(f"Early stopping at epoch {epoch + 1}")
            break
def evaluate(model, data_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in data_loader:
            out = model(data)
            pred = out.argmax(dim=1)
            y_true.extend(data.y.tolist())
            y_pred.extend(pred.tolist())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return acc, prec, rec, f1