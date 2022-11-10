import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from MIL_model import MIL_model
from MIL_data_loader import BagDataset, collate_bag_batches


device = "cuda:7"

data_path = "/home/jleiby/abdominal_ct/data/encoder_only/overlap_data/"


md = pd.read_csv("/home/jleiby/abdominal_ct/data/text_files/ct_labeled_data.csv", names=['ID', 'label'])
files = md.ID.tolist()
labels = md.label.tolist()


bag_data = BagDataset(data_path = data_path,
                     files = files,
                     labels = labels)

dl = DataLoader(bag_data, batch_size=10, shuffle=True, collate_fn=collate_bag_batches)

mod = MIL_model().to(device)
opt = torch.optim.Adam(mod.parameters())
#opt = torch.optim.SGD(mod.parameters(), lr=0.001, weight_decay=0.005)
criterion = nn.BCELoss().to(device)

train_loss = []

for j in range(20):
    mod.train()
    for i, (dat, lab) in enumerate(dl):
        opt.zero_grad()
        b_out = torch.empty(0).to(device)
        for d in dat:
            d = d.to(device)
            out, A = mod(d)
            b_out = torch.cat((b_out, out))
        b_out = b_out.unsqueeze(1)
        lab = torch.stack(lab).unsqueeze(1).float().to(device)
        loss = criterion(b_out, lab)
        loss.backward()
        opt.step()
        train_loss.append(loss.item())
        # print(loss.item())
    # print(A)
    print("Loss: ", loss.item())

    # performance metrics
    all_labels = torch.empty(0).to(device)
    all_out = torch.empty(0).to(device)
    with torch.no_grad():
        mod.eval()
        for i, (dat, lab) in enumerate(dl):
            b_out = torch.empty(0).to(device)
            for d in dat:
                d = d.to(device)
                out, _ = mod(d)
                b_out = torch.cat((b_out, out))
            b_out = b_out.unsqueeze(1)
            lab = torch.stack(lab).unsqueeze(1).float().to(device)

            all_labels = torch.cat((all_labels, lab), 0)
            all_out = torch.cat((all_out, b_out), 0)
    auc = roc_auc_score(all_labels.detach().cpu().numpy(), all_out.detach().cpu().numpy())
    print(f"AUC: {auc:.5f}")
    


# print(train_loss)