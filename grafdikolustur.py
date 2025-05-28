# Re-import necessary packages due to kernel reset
import matplotlib.pyplot as plt
import pandas as pd
import re

# Define the parser again
def parse_metrics(data, config_name):
    epochs, losses, accs, f1s = [], [], [], []
    l_match = re.search(r"L(\d+)", config_name)
    h_match = re.search(r"H(\d+)", config_name)
    num_layers = int(l_match.group(1)) if l_match else None
    hidden_channels = int(h_match.group(1)) if h_match else None

    for line in data.strip().split('\n'):
        match = re.match(r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Val Acc:\s+([\d.]+)\s+\|\s+F1:\s+([\d.]+)", line)
        if match:
            epoch, loss, acc, f1 = map(float, match.groups())
            epochs.append(int(epoch))
            losses.append(loss)
            accs.append(acc)
            f1s.append(f1)
    return pd.DataFrame({
        'Epoch': epochs,
        'Loss': losses,
        'Accuracy': accs,
        'F1 Score': f1s,
        'Config': config_name,
        'Layers': num_layers,
        'Hidden Channels': hidden_channels
    })

# Define function to plot the metric impact
def plot_metric_impact(df, var, metric):
    plt.figure(figsize=(10, 5))
    for value in sorted(df[var].unique()):
        group = df[df[var] == value]
        avg_by_epoch = group.groupby('Epoch')[metric].mean()
        plt.plot(avg_by_epoch.index, avg_by_epoch.values, label=f"{var}={value}")
    plt.title(f"{metric} vs Epoch for different {var} values")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# Data reloading due to session reset — re-load a small test set
configurations = {
    "L4_H128": """Epoch 01 | Loss: 508.4728 | Val Acc: 0.2772 | F1: 0.1048
Epoch 02 | Loss: 473.4315 | Val Acc: 0.3796 | F1: 0.2554
Epoch 03 | Loss: 438.4665 | Val Acc: 0.4297 | F1: 0.3096
Epoch 04 | Loss: 402.0264 | Val Acc: 0.5047 | F1: 0.3953
Epoch 05 | Loss: 368.2973 | Val Acc: 0.5585 | F1: 0.4837
Epoch 06 | Loss: 339.8988 | Val Acc: 0.5723 | F1: 0.4967
Epoch 07 | Loss: 312.1171 | Val Acc: 0.6275 | F1: 0.5364
Epoch 08 | Loss: 292.3211 | Val Acc: 0.6235 | F1: 0.5549
Epoch 09 | Loss: 276.1815 | Val Acc: 0.6727 | F1: 0.5938
Epoch 10 | Loss: 259.8830 | Val Acc: 0.6819 | F1: 0.6146
Epoch 11 | Loss: 250.5469 | Val Acc: 0.7003 | F1: 0.6263
Epoch 12 | Loss: 241.2677 | Val Acc: 0.7160 | F1: 0.6394
Epoch 13 | Loss: 229.8787 | Val Acc: 0.7118 | F1: 0.6421
Epoch 14 | Loss: 222.0105 | Val Acc: 0.7454 | F1: 0.6828
Epoch 15 | Loss: 214.1691 | Val Acc: 0.7325 | F1: 0.6687
Epoch 16 | Loss: 205.9956 | Val Acc: 0.7439 | F1: 0.6936
Epoch 17 | Loss: 200.5607 | Val Acc: 0.7780 | F1: 0.7267
Epoch 18 | Loss: 195.4208 | Val Acc: 0.7732 | F1: 0.7231
Epoch 19 | Loss: 191.2976 | Val Acc: 0.7707 | F1: 0.7220
Epoch 20 | Loss: 184.4078 | Val Acc: 0.7868 | F1: 0.7398
Epoch 21 | Loss: 180.0042 | Val Acc: 0.7940 | F1: 0.7532
Epoch 22 | Loss: 176.9452 | Val Acc: 0.8007 | F1: 0.7605
Epoch 23 | Loss: 174.3465 | Val Acc: 0.7924 | F1: 0.7540
Epoch 24 | Loss: 168.7827 | Val Acc: 0.7977 | F1: 0.7566
Epoch 25 | Loss: 166.0685 | Val Acc: 0.8038 | F1: 0.7646
Epoch 26 | Loss: 163.9936 | Val Acc: 0.8114 | F1: 0.7707
Epoch 27 | Loss: 161.8436 | Val Acc: 0.8131 | F1: 0.7743
Epoch 28 | Loss: 156.6154 | Val Acc: 0.8114 | F1: 0.7793
Epoch 29 | Loss: 154.2639 | Val Acc: 0.8203 | F1: 0.7856
Epoch 30 | Loss: 152.9516 | Val Acc: 0.8157 | F1: 0.7824
Epoch 31 | Loss: 149.8169 | Val Acc: 0.8208 | F1: 0.7884
Epoch 32 | Loss: 147.5242 | Val Acc: 0.8290 | F1: 0.7965
Epoch 33 | Loss: 145.0219 | Val Acc: 0.8280 | F1: 0.7991
Epoch 34 | Loss: 142.4777 | Val Acc: 0.8210 | F1: 0.7857
Epoch 35 | Loss: 141.4933 | Val Acc: 0.8303 | F1: 0.7985
Epoch 36 | Loss: 140.7229 | Val Acc: 0.8301 | F1: 0.8005
Epoch 37 | Loss: 138.1803 | Val Acc: 0.8350 | F1: 0.8048
Epoch 38 | Loss: 138.2799 | Val Acc: 0.8270 | F1: 0.7955
Epoch 39 | Loss: 135.3057 | Val Acc: 0.8349 | F1: 0.8073
Epoch 40 | Loss: 132.4903 | Val Acc: 0.8290 | F1: 0.7978
Epoch 41 | Loss: 128.9999 | Val Acc: 0.8401 | F1: 0.8111
Epoch 42 | Loss: 130.1105 | Val Acc: 0.8416 | F1: 0.8111
Epoch 43 | Loss: 130.3525 | Val Acc: 0.8213 | F1: 0.7934
Epoch 44 | Loss: 129.7122 | Val Acc: 0.8407 | F1: 0.8104
Epoch 45 | Loss: 127.4467 | Val Acc: 0.8429 | F1: 0.8091
Epoch 46 | Loss: 125.5206 | Val Acc: 0.8324 | F1: 0.8050

""",
    "L4_H256": """Epoch 01 | Loss: 504.3870 | Val Acc: 0.3276 | F1: 0.1805
Epoch 02 | Loss: 459.0278 | Val Acc: 0.4121 | F1: 0.2829
Epoch 03 | Loss: 410.9833 | Val Acc: 0.5004 | F1: 0.3991
Epoch 04 | Loss: 372.5794 | Val Acc: 0.5462 | F1: 0.4760
Epoch 05 | Loss: 335.4784 | Val Acc: 0.6041 | F1: 0.5385
Epoch 06 | Loss: 305.4937 | Val Acc: 0.6418 | F1: 0.5674
Epoch 07 | Loss: 280.7685 | Val Acc: 0.6716 | F1: 0.6133
Epoch 08 | Loss: 262.0666 | Val Acc: 0.6926 | F1: 0.6246
Epoch 09 | Loss: 245.7318 | Val Acc: 0.7042 | F1: 0.6479
Epoch 10 | Loss: 231.5697 | Val Acc: 0.7199 | F1: 0.6621
Epoch 11 | Loss: 222.3236 | Val Acc: 0.7402 | F1: 0.6883
Epoch 12 | Loss: 212.1063 | Val Acc: 0.7466 | F1: 0.6994
Epoch 13 | Loss: 200.8836 | Val Acc: 0.7645 | F1: 0.7162
Epoch 14 | Loss: 193.0506 | Val Acc: 0.7838 | F1: 0.7368
Epoch 15 | Loss: 185.2512 | Val Acc: 0.7758 | F1: 0.7375
Epoch 16 | Loss: 180.1048 | Val Acc: 0.7887 | F1: 0.7451
Epoch 17 | Loss: 177.3904 | Val Acc: 0.7998 | F1: 0.7642
Epoch 18 | Loss: 168.3850 | Val Acc: 0.8030 | F1: 0.7690
Epoch 19 | Loss: 164.3292 | Val Acc: 0.7945 | F1: 0.7614
Epoch 20 | Loss: 162.2249 | Val Acc: 0.8058 | F1: 0.7655
Epoch 21 | Loss: 156.4635 | Val Acc: 0.8134 | F1: 0.7764
Epoch 22 | Loss: 150.3090 | Val Acc: 0.8190 | F1: 0.7864
Epoch 23 | Loss: 147.6787 | Val Acc: 0.8271 | F1: 0.7955
Epoch 24 | Loss: 146.9755 | Val Acc: 0.8241 | F1: 0.7915
Epoch 25 | Loss: 142.6544 | Val Acc: 0.8251 | F1: 0.7925
Epoch 26 | Loss: 139.8235 | Val Acc: 0.8231 | F1: 0.7920
Epoch 27 | Loss: 136.6894 | Val Acc: 0.8254 | F1: 0.7851
Epoch 28 | Loss: 136.4850 | Val Acc: 0.8324 | F1: 0.8019
Epoch 29 | Loss: 133.3528 | Val Acc: 0.8360 | F1: 0.8047
Epoch 30 | Loss: 129.2370 | Val Acc: 0.8344 | F1: 0.8039
Epoch 31 | Loss: 128.1795 | Val Acc: 0.8449 | F1: 0.8162
Epoch 32 | Loss: 125.9170 | Val Acc: 0.8394 | F1: 0.8113
Epoch 33 | Loss: 125.0805 | Val Acc: 0.8416 | F1: 0.8075
Epoch 34 | Loss: 122.6266 | Val Acc: 0.8310 | F1: 0.7989
Epoch 35 | Loss: 121.9680 | Val Acc: 0.8442 | F1: 0.8108
Epoch 36 | Loss: 120.0096 | Val Acc: 0.8440 | F1: 0.8141

""",
    "L4_H512": """Epoch 01 | Loss: 498.6898 | Val Acc: 0.3181 | F1: 0.1728
Epoch 02 | Loss: 442.5225 | Val Acc: 0.4585 | F1: 0.3379
Epoch 03 | Loss: 391.3260 | Val Acc: 0.5320 | F1: 0.4350
Epoch 04 | Loss: 345.2095 | Val Acc: 0.6028 | F1: 0.5201
Epoch 05 | Loss: 307.6465 | Val Acc: 0.6268 | F1: 0.5600
Epoch 06 | Loss: 276.9920 | Val Acc: 0.6797 | F1: 0.6135
Epoch 07 | Loss: 251.1981 | Val Acc: 0.6937 | F1: 0.6340
Epoch 08 | Loss: 234.0571 | Val Acc: 0.6824 | F1: 0.6251
Epoch 09 | Loss: 216.9723 | Val Acc: 0.7372 | F1: 0.6854
Epoch 10 | Loss: 203.5470 | Val Acc: 0.7624 | F1: 0.7052
Epoch 11 | Loss: 190.5767 | Val Acc: 0.7677 | F1: 0.7230
Epoch 12 | Loss: 181.5603 | Val Acc: 0.7905 | F1: 0.7443
Epoch 13 | Loss: 170.7067 | Val Acc: 0.7994 | F1: 0.7583
Epoch 14 | Loss: 165.3514 | Val Acc: 0.7878 | F1: 0.7491
Epoch 15 | Loss: 160.4909 | Val Acc: 0.8084 | F1: 0.7730
Epoch 16 | Loss: 153.7202 | Val Acc: 0.8178 | F1: 0.7831
Epoch 17 | Loss: 150.1390 | Val Acc: 0.8148 | F1: 0.7840
Epoch 18 | Loss: 143.9314 | Val Acc: 0.8198 | F1: 0.7835
Epoch 19 | Loss: 138.7657 | Val Acc: 0.8188 | F1: 0.7876
Epoch 20 | Loss: 135.5790 | Val Acc: 0.8346 | F1: 0.8071
Epoch 21 | Loss: 132.9484 | Val Acc: 0.8204 | F1: 0.7923
Epoch 22 | Loss: 130.5562 | Val Acc: 0.8227 | F1: 0.7898
Epoch 23 | Loss: 127.2175 | Val Acc: 0.8389 | F1: 0.8082
Epoch 24 | Loss: 124.9985 | Val Acc: 0.8413 | F1: 0.8132
Epoch 25 | Loss: 122.9997 | Val Acc: 0.8396 | F1: 0.8122
Epoch 26 | Loss: 120.3390 | Val Acc: 0.8411 | F1: 0.8169
Epoch 27 | Loss: 118.8076 | Val Acc: 0.8497 | F1: 0.8243
Epoch 28 | Loss: 116.0834 | Val Acc: 0.8487 | F1: 0.8211
Epoch 29 | Loss: 113.9194 | Val Acc: 0.8452 | F1: 0.8182
Epoch 30 | Loss: 112.0125 | Val Acc: 0.8477 | F1: 0.8221
Epoch 31 | Loss: 111.4042 | Val Acc: 0.8569 | F1: 0.8305
Epoch 32 | Loss: 110.5681 | Val Acc: 0.8493 | F1: 0.8222
Epoch 33 | Loss: 107.9115 | Val Acc: 0.8592 | F1: 0.8322
Epoch 34 | Loss: 105.7466 | Val Acc: 0.8549 | F1: 0.8298
Epoch 35 | Loss: 105.2362 | Val Acc: 0.8599 | F1: 0.8332
Epoch 36 | Loss: 103.8668 | Val Acc: 0.8564 | F1: 0.8338
Epoch 37 | Loss: 101.9013 | Val Acc: 0.8567 | F1: 0.8285
Epoch 38 | Loss: 99.8982 | Val Acc: 0.8629 | F1: 0.8390
Epoch 39 | Loss: 100.9848 | Val Acc: 0.8557 | F1: 0.8284
Epoch 40 | Loss: 99.4479 | Val Acc: 0.8566 | F1: 0.8282
Epoch 41 | Loss: 98.5284 | Val Acc: 0.8534 | F1: 0.8287
Epoch 42 | Loss: 96.4166 | Val Acc: 0.8597 | F1: 0.8361
Epoch 43 | Loss: 96.6248 | Val Acc: 0.8626 | F1: 0.8388


""",
    "L4_H1024": """Epoch 01 | Loss: 514.0366 | Val Acc: 0.2858 | F1: 0.1418
Epoch 02 | Loss: 464.1474 | Val Acc: 0.4099 | F1: 0.2733
Epoch 03 | Loss: 418.4323 | Val Acc: 0.4537 | F1: 0.3289
Epoch 04 | Loss: 377.8182 | Val Acc: 0.5363 | F1: 0.4524
Epoch 05 | Loss: 344.7648 | Val Acc: 0.5801 | F1: 0.5101
Epoch 06 | Loss: 318.7748 | Val Acc: 0.6134 | F1: 0.5540
Epoch 07 | Loss: 295.8095 | Val Acc: 0.6480 | F1: 0.5803
Epoch 08 | Loss: 276.8379 | Val Acc: 0.6870 | F1: 0.6146
Epoch 09 | Loss: 260.8992 | Val Acc: 0.7012 | F1: 0.6377
Epoch 10 | Loss: 244.6511 | Val Acc: 0.7076 | F1: 0.6444
Epoch 11 | Loss: 231.4817 | Val Acc: 0.7205 | F1: 0.6644
Epoch 12 | Loss: 219.8074 | Val Acc: 0.7582 | F1: 0.7063
Epoch 13 | Loss: 208.7422 | Val Acc: 0.7508 | F1: 0.6986
Epoch 14 | Loss: 200.2132 | Val Acc: 0.7705 | F1: 0.7228
Epoch 15 | Loss: 191.0892 | Val Acc: 0.7880 | F1: 0.7462
Epoch 16 | Loss: 185.2499 | Val Acc: 0.7877 | F1: 0.7401
Epoch 17 | Loss: 177.7060 | Val Acc: 0.7920 | F1: 0.7462
Epoch 18 | Loss: 171.9207 | Val Acc: 0.7948 | F1: 0.7556
Epoch 19 | Loss: 166.6242 | Val Acc: 0.7943 | F1: 0.7590
Epoch 20 | Loss: 162.4155 | Val Acc: 0.7955 | F1: 0.7589
Epoch 21 | Loss: 156.1841 | Val Acc: 0.8144 | F1: 0.7789
Epoch 22 | Loss: 153.3302 | Val Acc: 0.8147 | F1: 0.7819
Epoch 23 | Loss: 148.3658 | Val Acc: 0.8200 | F1: 0.7850
Epoch 24 | Loss: 145.3765 | Val Acc: 0.8151 | F1: 0.7827
Epoch 25 | Loss: 142.4269 | Val Acc: 0.8243 | F1: 0.7884
Epoch 26 | Loss: 139.9210 | Val Acc: 0.8243 | F1: 0.7914
Epoch 27 | Loss: 136.7725 | Val Acc: 0.8193 | F1: 0.7869
Epoch 28 | Loss: 135.6063 | Val Acc: 0.8334 | F1: 0.7992
Epoch 29 | Loss: 132.3137 | Val Acc: 0.8380 | F1: 0.8094
Epoch 30 | Loss: 131.7062 | Val Acc: 0.8304 | F1: 0.8005
Epoch 31 | Loss: 128.1170 | Val Acc: 0.8357 | F1: 0.8027
Epoch 32 | Loss: 127.2173 | Val Acc: 0.8277 | F1: 0.7959
Epoch 33 | Loss: 124.9621 | Val Acc: 0.8387 | F1: 0.8072
Epoch 34 | Loss: 121.5255 | Val Acc: 0.8423 | F1: 0.8150
Epoch 35 | Loss: 120.8233 | Val Acc: 0.8460 | F1: 0.8177
Epoch 36 | Loss: 120.6265 | Val Acc: 0.8390 | F1: 0.8045
Epoch 37 | Loss: 117.1514 | Val Acc: 0.8490 | F1: 0.8185
Epoch 38 | Loss: 118.0680 | Val Acc: 0.8509 | F1: 0.8215
Epoch 39 | Loss: 115.7765 | Val Acc: 0.8420 | F1: 0.8130
Epoch 40 | Loss: 114.2782 | Val Acc: 0.8532 | F1: 0.8237
Epoch 41 | Loss: 113.4211 | Val Acc: 0.8484 | F1: 0.8188
Epoch 42 | Loss: 112.4896 | Val Acc: 0.8520 | F1: 0.8255
Epoch 43 | Loss: 111.4161 | Val Acc: 0.8462 | F1: 0.8191
Epoch 44 | Loss: 109.7653 | Val Acc: 0.8466 | F1: 0.8174
Epoch 45 | Loss: 107.6834 | Val Acc: 0.8550 | F1: 0.8281
Epoch 46 | Loss: 108.2021 | Val Acc: 0.8516 | F1: 0.8223
Epoch 47 | Loss: 107.7804 | Val Acc: 0.8459 | F1: 0.8191
Epoch 48 | Loss: 106.6931 | Val Acc: 0.8564 | F1: 0.8298
Epoch 49 | Loss: 104.0744 | Val Acc: 0.8526 | F1: 0.8239
Epoch 50 | Loss: 104.1905 | Val Acc: 0.8606 | F1: 0.8274
Epoch 51 | Loss: 102.4697 | Val Acc: 0.8637 | F1: 0.8381
Epoch 52 | Loss: 102.7734 | Val Acc: 0.8540 | F1: 0.8263
Epoch 53 | Loss: 101.7635 | Val Acc: 0.8593 | F1: 0.8288
Epoch 54 | Loss: 101.0312 | Val Acc: 0.8612 | F1: 0.8378
Epoch 55 | Loss: 101.1521 | Val Acc: 0.8605 | F1: 0.8366
Epoch 56 | Loss: 100.4126 | Val Acc: 0.8577 | F1: 0.8287

""",
    "L5_H128": """Epoch 01 | Loss: 504.3602 | Val Acc: 0.2961 | F1: 0.1548
Epoch 02 | Loss: 463.7956 | Val Acc: 0.3985 | F1: 0.2793
Epoch 03 | Loss: 420.5634 | Val Acc: 0.4527 | F1: 0.3275
Epoch 04 | Loss: 383.1836 | Val Acc: 0.5320 | F1: 0.4362
Epoch 05 | Loss: 353.5648 | Val Acc: 0.5366 | F1: 0.4620
Epoch 06 | Loss: 324.6396 | Val Acc: 0.6009 | F1: 0.5112
Epoch 07 | Loss: 300.3097 | Val Acc: 0.6385 | F1: 0.5639
Epoch 08 | Loss: 283.8878 | Val Acc: 0.6629 | F1: 0.5980
Epoch 09 | Loss: 268.3013 | Val Acc: 0.6863 | F1: 0.6156
Epoch 10 | Loss: 255.6899 | Val Acc: 0.7082 | F1: 0.6410
Epoch 11 | Loss: 243.9425 | Val Acc: 0.7073 | F1: 0.6486
Epoch 12 | Loss: 233.2478 | Val Acc: 0.6979 | F1: 0.6366
Epoch 13 | Loss: 223.5956 | Val Acc: 0.7345 | F1: 0.6833
Epoch 14 | Loss: 214.5556 | Val Acc: 0.7216 | F1: 0.6660
Epoch 15 | Loss: 207.0085 | Val Acc: 0.7488 | F1: 0.7019
Epoch 16 | Loss: 200.3348 | Val Acc: 0.7609 | F1: 0.7157
Epoch 17 | Loss: 194.4193 | Val Acc: 0.7621 | F1: 0.7190
Epoch 18 | Loss: 188.1596 | Val Acc: 0.7684 | F1: 0.7220
Epoch 19 | Loss: 182.8828 | Val Acc: 0.7812 | F1: 0.7413
Epoch 20 | Loss: 177.3207 | Val Acc: 0.7864 | F1: 0.7411
Epoch 21 | Loss: 173.0918 | Val Acc: 0.7933 | F1: 0.7539
Epoch 22 | Loss: 169.4529 | Val Acc: 0.7921 | F1: 0.7506
Epoch 23 | Loss: 165.7912 | Val Acc: 0.8091 | F1: 0.7720
Epoch 24 | Loss: 161.2366 | Val Acc: 0.8074 | F1: 0.7711
Epoch 25 | Loss: 160.7642 | Val Acc: 0.8017 | F1: 0.7673
Epoch 26 | Loss: 155.9202 | Val Acc: 0.8163 | F1: 0.7786
Epoch 27 | Loss: 156.8628 | Val Acc: 0.8088 | F1: 0.7773
Epoch 28 | Loss: 148.4374 | Val Acc: 0.8238 | F1: 0.7902
Epoch 29 | Loss: 148.1187 | Val Acc: 0.8151 | F1: 0.7793
Epoch 30 | Loss: 145.6943 | Val Acc: 0.8250 | F1: 0.7875
Epoch 31 | Loss: 143.2348 | Val Acc: 0.8327 | F1: 0.7987
Epoch 32 | Loss: 141.5017 | Val Acc: 0.8253 | F1: 0.7942
Epoch 33 | Loss: 138.2197 | Val Acc: 0.8250 | F1: 0.7929
Epoch 34 | Loss: 139.0167 | Val Acc: 0.8238 | F1: 0.7905
Epoch 35 | Loss: 134.2810 | Val Acc: 0.8299 | F1: 0.7997
Epoch 36 | Loss: 133.6444 | Val Acc: 0.8274 | F1: 0.7985
Epoch 37 | Loss: 131.5681 | Val Acc: 0.8279 | F1: 0.7991
Epoch 38 | Loss: 133.4420 | Val Acc: 0.8227 | F1: 0.7898
Epoch 39 | Loss: 127.2976 | Val Acc: 0.8430 | F1: 0.8150
Epoch 40 | Loss: 126.6290 | Val Acc: 0.8281 | F1: 0.7969
Epoch 41 | Loss: 126.7421 | Val Acc: 0.8396 | F1: 0.8105
Epoch 42 | Loss: 126.3955 | Val Acc: 0.8432 | F1: 0.8149
Epoch 43 | Loss: 124.6823 | Val Acc: 0.8256 | F1: 0.7990
Epoch 44 | Loss: 122.8247 | Val Acc: 0.8410 | F1: 0.8149
""",
    "L5_H256": """Epoch 01 | Loss: 502.0042 | Val Acc: 0.3073 | F1: 0.1927
Epoch 02 | Loss: 451.6561 | Val Acc: 0.4245 | F1: 0.2993
Epoch 03 | Loss: 402.1237 | Val Acc: 0.4831 | F1: 0.3842
Epoch 04 | Loss: 357.8086 | Val Acc: 0.5708 | F1: 0.4931
Epoch 05 | Loss: 318.9313 | Val Acc: 0.6278 | F1: 0.5425
Epoch 06 | Loss: 285.4365 | Val Acc: 0.6571 | F1: 0.5849
Epoch 07 | Loss: 259.6344 | Val Acc: 0.6882 | F1: 0.6137
Epoch 08 | Loss: 240.4831 | Val Acc: 0.7145 | F1: 0.6552
Epoch 09 | Loss: 221.9912 | Val Acc: 0.7359 | F1: 0.6862
Epoch 10 | Loss: 207.8591 | Val Acc: 0.7562 | F1: 0.7025
Epoch 11 | Loss: 195.2939 | Val Acc: 0.7678 | F1: 0.7198
Epoch 12 | Loss: 186.8244 | Val Acc: 0.7721 | F1: 0.7282
Epoch 13 | Loss: 177.9179 | Val Acc: 0.7900 | F1: 0.7542
Epoch 14 | Loss: 169.9135 | Val Acc: 0.7971 | F1: 0.7610
Epoch 15 | Loss: 163.3666 | Val Acc: 0.8015 | F1: 0.7671
Epoch 16 | Loss: 156.8959 | Val Acc: 0.7977 | F1: 0.7549
Epoch 17 | Loss: 151.8691 | Val Acc: 0.8091 | F1: 0.7748
Epoch 18 | Loss: 147.6883 | Val Acc: 0.8063 | F1: 0.7749
Epoch 19 | Loss: 142.8980 | Val Acc: 0.8181 | F1: 0.7840
Epoch 20 | Loss: 139.7173 | Val Acc: 0.8289 | F1: 0.7965
Epoch 21 | Loss: 137.4318 | Val Acc: 0.8321 | F1: 0.7988
Epoch 22 | Loss: 132.5488 | Val Acc: 0.8270 | F1: 0.7990
Epoch 23 | Loss: 131.6496 | Val Acc: 0.8333 | F1: 0.8013
Epoch 24 | Loss: 130.0217 | Val Acc: 0.8364 | F1: 0.8122
Epoch 25 | Loss: 123.7512 | Val Acc: 0.8394 | F1: 0.8127
Epoch 26 | Loss: 122.7846 | Val Acc: 0.8434 | F1: 0.8128
Epoch 27 | Loss: 120.7484 | Val Acc: 0.8273 | F1: 0.7955
Epoch 28 | Loss: 119.6955 | Val Acc: 0.8394 | F1: 0.8055
Epoch 29 | Loss: 116.7968 | Val Acc: 0.8462 | F1: 0.8109
Epoch 30 | Loss: 114.2617 | Val Acc: 0.8489 | F1: 0.8222
Epoch 31 | Loss: 113.1645 | Val Acc: 0.8463 | F1: 0.8171
Epoch 32 | Loss: 112.2470 | Val Acc: 0.8444 | F1: 0.8167
Epoch 33 | Loss: 109.9536 | Val Acc: 0.8387 | F1: 0.8133
Epoch 34 | Loss: 109.1547 | Val Acc: 0.8502 | F1: 0.8213
Epoch 35 | Loss: 107.6248 | Val Acc: 0.8513 | F1: 0.8238
Epoch 36 | Loss: 104.6365 | Val Acc: 0.8553 | F1: 0.8305
Epoch 37 | Loss: 105.0469 | Val Acc: 0.8436 | F1: 0.8155
Epoch 38 | Loss: 103.7896 | Val Acc: 0.8579 | F1: 0.8321
Epoch 39 | Loss: 101.2225 | Val Acc: 0.8566 | F1: 0.8306
Epoch 40 | Loss: 101.5574 | Val Acc: 0.8556 | F1: 0.8295
Epoch 41 | Loss: 99.6048 | Val Acc: 0.8570 | F1: 0.8336
Epoch 42 | Loss: 99.3457 | Val Acc: 0.8549 | F1: 0.8245
Epoch 43 | Loss: 98.8860 | Val Acc: 0.8573 | F1: 0.8305
Epoch 44 | Loss: 95.9099 | Val Acc: 0.8607 | F1: 0.8378
Epoch 45 | Loss: 98.3086 | Val Acc: 0.8580 | F1: 0.8330
Epoch 46 | Loss: 94.7830 | Val Acc: 0.8577 | F1: 0.8344
Epoch 47 | Loss: 94.2854 | Val Acc: 0.8653 | F1: 0.8424
Epoch 48 | Loss: 92.7391 | Val Acc: 0.8665 | F1: 0.8433
Epoch 49 | Loss: 92.6132 | Val Acc: 0.8619 | F1: 0.8360
Epoch 50 | Loss: 92.3392 | Val Acc: 0.8666 | F1: 0.8411
Epoch 51 | Loss: 90.9637 | Val Acc: 0.8609 | F1: 0.8320
Epoch 52 | Loss: 91.1380 | Val Acc: 0.8606 | F1: 0.8361
Epoch 53 | Loss: 89.0172 | Val Acc: 0.8693 | F1: 0.8465
Epoch 54 | Loss: 88.9182 | Val Acc: 0.8667 | F1: 0.8435
Epoch 55 | Loss: 90.8819 | Val Acc: 0.8599 | F1: 0.8344
Epoch 56 | Loss: 87.2530 | Val Acc: 0.8710 | F1: 0.8493
Epoch 57 | Loss: 86.3727 | Val Acc: 0.8613 | F1: 0.8318
Epoch 58 | Loss: 87.3347 | Val Acc: 0.8623 | F1: 0.8404
Epoch 59 | Loss: 86.6064 | Val Acc: 0.8643 | F1: 0.8399
Epoch 60 | Loss: 86.6551 | Val Acc: 0.8613 | F1: 0.8423
Epoch 61 | Loss: 87.5252 | Val Acc: 0.8702 | F1: 0.8468
""",
    "L5_H512": """Epoch 01 | Loss: 507.1274 | Val Acc: 0.2927 | F1: 0.1414
Epoch 02 | Loss: 457.2155 | Val Acc: 0.4121 | F1: 0.2621
Epoch 03 | Loss: 410.9466 | Val Acc: 0.4520 | F1: 0.3413
Epoch 04 | Loss: 368.5790 | Val Acc: 0.5639 | F1: 0.4927
Epoch 05 | Loss: 328.9478 | Val Acc: 0.6164 | F1: 0.5458
Epoch 06 | Loss: 296.7125 | Val Acc: 0.6430 | F1: 0.5743
Epoch 07 | Loss: 271.1742 | Val Acc: 0.6800 | F1: 0.6193
Epoch 08 | Loss: 251.3792 | Val Acc: 0.6854 | F1: 0.6220
Epoch 09 | Loss: 234.3630 | Val Acc: 0.7153 | F1: 0.6593
Epoch 10 | Loss: 219.1034 | Val Acc: 0.7469 | F1: 0.6966
Epoch 11 | Loss: 205.5524 | Val Acc: 0.7434 | F1: 0.6990
Epoch 12 | Loss: 196.7565 | Val Acc: 0.7522 | F1: 0.7131
Epoch 13 | Loss: 185.8578 | Val Acc: 0.7807 | F1: 0.7395
Epoch 14 | Loss: 175.7178 | Val Acc: 0.7941 | F1: 0.7554
Epoch 15 | Loss: 171.1688 | Val Acc: 0.7842 | F1: 0.7494
Epoch 16 | Loss: 164.1127 | Val Acc: 0.7971 | F1: 0.7560
Epoch 17 | Loss: 157.9440 | Val Acc: 0.8101 | F1: 0.7797
Epoch 18 | Loss: 152.1515 | Val Acc: 0.8167 | F1: 0.7866
Epoch 19 | Loss: 146.6216 | Val Acc: 0.8088 | F1: 0.7755
Epoch 20 | Loss: 144.4648 | Val Acc: 0.8279 | F1: 0.7955
Epoch 21 | Loss: 141.5618 | Val Acc: 0.8256 | F1: 0.7950
Epoch 22 | Loss: 137.1726 | Val Acc: 0.8350 | F1: 0.8055
Epoch 23 | Loss: 133.8316 | Val Acc: 0.8374 | F1: 0.8076
Epoch 24 | Loss: 129.1824 | Val Acc: 0.8150 | F1: 0.7831
Epoch 25 | Loss: 128.6243 | Val Acc: 0.8386 | F1: 0.8116
Epoch 26 | Loss: 126.2851 | Val Acc: 0.8403 | F1: 0.8106
Epoch 27 | Loss: 122.3890 | Val Acc: 0.8493 | F1: 0.8232
Epoch 28 | Loss: 119.8820 | Val Acc: 0.8373 | F1: 0.8091
Epoch 29 | Loss: 117.1553 | Val Acc: 0.8459 | F1: 0.8189
Epoch 30 | Loss: 115.8386 | Val Acc: 0.8477 | F1: 0.8216
Epoch 31 | Loss: 114.3447 | Val Acc: 0.8529 | F1: 0.8265
Epoch 32 | Loss: 114.0841 | Val Acc: 0.8499 | F1: 0.8230
Epoch 33 | Loss: 111.4418 | Val Acc: 0.8572 | F1: 0.8337
Epoch 34 | Loss: 109.3273 | Val Acc: 0.8544 | F1: 0.8297
Epoch 35 | Loss: 107.4315 | Val Acc: 0.8557 | F1: 0.8275
Epoch 36 | Loss: 105.2575 | Val Acc: 0.8503 | F1: 0.8237
Epoch 37 | Loss: 105.1011 | Val Acc: 0.8585 | F1: 0.8357
Epoch 38 | Loss: 102.5548 | Val Acc: 0.8582 | F1: 0.8345
Epoch 39 | Loss: 101.9994 | Val Acc: 0.8576 | F1: 0.8306
Epoch 40 | Loss: 102.2515 | Val Acc: 0.8560 | F1: 0.8271
Epoch 41 | Loss: 99.1280 | Val Acc: 0.8617 | F1: 0.8397
Epoch 42 | Loss: 100.7821 | Val Acc: 0.8599 | F1: 0.8337
Epoch 43 | Loss: 97.6146 | Val Acc: 0.8592 | F1: 0.8340
Epoch 44 | Loss: 96.2373 | Val Acc: 0.8577 | F1: 0.8305
Epoch 45 | Loss: 95.6098 | Val Acc: 0.8600 | F1: 0.8365
Epoch 46 | Loss: 95.0596 | Val Acc: 0.8647 | F1: 0.8408
Epoch 47 | Loss: 95.2202 | Val Acc: 0.8689 | F1: 0.8453
Epoch 48 | Loss: 92.7495 | Val Acc: 0.8629 | F1: 0.8364
Epoch 49 | Loss: 92.1071 | Val Acc: 0.8675 | F1: 0.8445
Epoch 50 | Loss: 91.9857 | Val Acc: 0.8579 | F1: 0.8332
Epoch 51 | Loss: 90.6000 | Val Acc: 0.8655 | F1: 0.8421
Epoch 52 | Loss: 89.2823 | Val Acc: 0.8722 | F1: 0.8484
Epoch 53 | Loss: 91.3482 | Val Acc: 0.8693 | F1: 0.8452
Epoch 54 | Loss: 88.4921 | Val Acc: 0.8672 | F1: 0.8439
Epoch 55 | Loss: 88.7395 | Val Acc: 0.8645 | F1: 0.8401
Epoch 56 | Loss: 86.6742 | Val Acc: 0.8622 | F1: 0.8398
Epoch 57 | Loss: 87.1374 | Val Acc: 0.8570 | F1: 0.8338
""",
    "L5_H1024": """Epoch 01 | Loss: 523.3421 | Val Acc: 0.2854 | F1: 0.1311
Epoch 02 | Loss: 469.4850 | Val Acc: 0.3589 | F1: 0.2085
Epoch 03 | Loss: 431.0713 | Val Acc: 0.4314 | F1: 0.3260
Epoch 04 | Loss: 395.3266 | Val Acc: 0.5013 | F1: 0.4368
Epoch 05 | Loss: 364.7256 | Val Acc: 0.5665 | F1: 0.4920
Epoch 06 | Loss: 333.9401 | Val Acc: 0.6008 | F1: 0.5304
Epoch 07 | Loss: 311.8505 | Val Acc: 0.6360 | F1: 0.5625
Epoch 08 | Loss: 289.0692 | Val Acc: 0.6418 | F1: 0.5766
Epoch 09 | Loss: 269.9942 | Val Acc: 0.6401 | F1: 0.5887
Epoch 10 | Loss: 254.9079 | Val Acc: 0.6929 | F1: 0.6440
Epoch 11 | Loss: 240.6683 | Val Acc: 0.7179 | F1: 0.6602
Epoch 12 | Loss: 229.8345 | Val Acc: 0.7253 | F1: 0.6713
Epoch 13 | Loss: 218.8546 | Val Acc: 0.7416 | F1: 0.6841
Epoch 14 | Loss: 211.5452 | Val Acc: 0.7612 | F1: 0.7153
Epoch 15 | Loss: 201.5090 | Val Acc: 0.7687 | F1: 0.7200
Epoch 16 | Loss: 194.2126 | Val Acc: 0.7665 | F1: 0.7254
Epoch 17 | Loss: 187.9196 | Val Acc: 0.7668 | F1: 0.7204
Epoch 18 | Loss: 184.5605 | Val Acc: 0.7971 | F1: 0.7545
Epoch 19 | Loss: 178.0670 | Val Acc: 0.7943 | F1: 0.7516
Epoch 20 | Loss: 171.8140 | Val Acc: 0.7910 | F1: 0.7507
Epoch 21 | Loss: 167.0012 | Val Acc: 0.7963 | F1: 0.7596
Epoch 22 | Loss: 164.2261 | Val Acc: 0.8034 | F1: 0.7644
Epoch 23 | Loss: 160.6415 | Val Acc: 0.8136 | F1: 0.7789
Epoch 24 | Loss: 155.7954 | Val Acc: 0.8048 | F1: 0.7636
Epoch 25 | Loss: 154.0409 | Val Acc: 0.8108 | F1: 0.7753
Epoch 26 | Loss: 149.5087 | Val Acc: 0.8068 | F1: 0.7724
Epoch 27 | Loss: 146.8561 | Val Acc: 0.8180 | F1: 0.7795
Epoch 28 | Loss: 143.6182 | Val Acc: 0.8134 | F1: 0.7782
Epoch 29 | Loss: 143.6210 | Val Acc: 0.8177 | F1: 0.7872
Epoch 30 | Loss: 139.1020 | Val Acc: 0.8250 | F1: 0.7892
Epoch 31 | Loss: 136.1060 | Val Acc: 0.8269 | F1: 0.7936
Epoch 32 | Loss: 134.0641 | Val Acc: 0.8406 | F1: 0.8058
Epoch 33 | Loss: 131.6294 | Val Acc: 0.8300 | F1: 0.7985
Epoch 34 | Loss: 133.1636 | Val Acc: 0.8234 | F1: 0.7903
Epoch 35 | Loss: 128.9753 | Val Acc: 0.8316 | F1: 0.7980
Epoch 36 | Loss: 125.8402 | Val Acc: 0.8306 | F1: 0.8008
Epoch 37 | Loss: 124.3124 | Val Acc: 0.8383 | F1: 0.8045
""",
    "L6_H128": """Epoch 01 | Loss: 507.9845 | Val Acc: 0.2661 | F1: 0.0831
Epoch 02 | Loss: 473.3107 | Val Acc: 0.3719 | F1: 0.2303
Epoch 03 | Loss: 433.1090 | Val Acc: 0.4404 | F1: 0.3365
Epoch 04 | Loss: 389.8586 | Val Acc: 0.5405 | F1: 0.4445
Epoch 05 | Loss: 357.4893 | Val Acc: 0.5729 | F1: 0.4855
Epoch 06 | Loss: 326.4079 | Val Acc: 0.6167 | F1: 0.5317
Epoch 07 | Loss: 300.2605 | Val Acc: 0.6274 | F1: 0.5598
Epoch 08 | Loss: 280.4946 | Val Acc: 0.6754 | F1: 0.6050
Epoch 09 | Loss: 257.4797 | Val Acc: 0.6760 | F1: 0.6052
Epoch 10 | Loss: 243.5531 | Val Acc: 0.7076 | F1: 0.6489
Epoch 11 | Loss: 227.8251 | Val Acc: 0.7202 | F1: 0.6551
Epoch 12 | Loss: 220.3439 | Val Acc: 0.7328 | F1: 0.6789
Epoch 13 | Loss: 208.4360 | Val Acc: 0.7509 | F1: 0.6979
Epoch 14 | Loss: 198.1464 | Val Acc: 0.7521 | F1: 0.7123
Epoch 15 | Loss: 191.8035 | Val Acc: 0.7757 | F1: 0.7324
Epoch 16 | Loss: 189.0368 | Val Acc: 0.7827 | F1: 0.7370
Epoch 17 | Loss: 179.5869 | Val Acc: 0.7841 | F1: 0.7449
Epoch 18 | Loss: 175.6706 | Val Acc: 0.7931 | F1: 0.7541
Epoch 19 | Loss: 167.8866 | Val Acc: 0.7881 | F1: 0.7526
Epoch 20 | Loss: 164.1017 | Val Acc: 0.8024 | F1: 0.7627
Epoch 21 | Loss: 160.1350 | Val Acc: 0.8031 | F1: 0.7673
Epoch 22 | Loss: 154.5416 | Val Acc: 0.8111 | F1: 0.7767
Epoch 23 | Loss: 154.5774 | Val Acc: 0.8078 | F1: 0.7724
Epoch 24 | Loss: 149.8423 | Val Acc: 0.8140 | F1: 0.7824
Epoch 25 | Loss: 149.6740 | Val Acc: 0.7968 | F1: 0.7639
Epoch 26 | Loss: 143.1272 | Val Acc: 0.8118 | F1: 0.7818
Epoch 27 | Loss: 143.9516 | Val Acc: 0.8254 | F1: 0.7949
Epoch 28 | Loss: 139.1626 | Val Acc: 0.8293 | F1: 0.7978
Epoch 29 | Loss: 135.3642 | Val Acc: 0.8231 | F1: 0.7954
Epoch 30 | Loss: 134.5557 | Val Acc: 0.8274 | F1: 0.7943
Epoch 31 | Loss: 133.0990 | Val Acc: 0.8359 | F1: 0.8067
Epoch 32 | Loss: 130.0841 | Val Acc: 0.8310 | F1: 0.8005
Epoch 33 | Loss: 131.4435 | Val Acc: 0.8264 | F1: 0.7963
Epoch 34 | Loss: 126.4885 | Val Acc: 0.8291 | F1: 0.8001
Epoch 35 | Loss: 125.7032 | Val Acc: 0.8310 | F1: 0.8026
Epoch 36 | Loss: 123.5780 | Val Acc: 0.8323 | F1: 0.8025
""",
    "L6_H256": """Epoch 01 | Loss: 503.5120 | Val Acc: 0.3178 | F1: 0.1520
Epoch 02 | Loss: 454.2024 | Val Acc: 0.4046 | F1: 0.2748
Epoch 03 | Loss: 398.4870 | Val Acc: 0.5036 | F1: 0.4006
Epoch 04 | Loss: 346.6598 | Val Acc: 0.5834 | F1: 0.5108
Epoch 05 | Loss: 305.2127 | Val Acc: 0.6304 | F1: 0.5670
Epoch 06 | Loss: 273.8081 | Val Acc: 0.6570 | F1: 0.6072
Epoch 07 | Loss: 250.3884 | Val Acc: 0.7120 | F1: 0.6571
Epoch 08 | Loss: 229.1880 | Val Acc: 0.7353 | F1: 0.6864
Epoch 09 | Loss: 214.8943 | Val Acc: 0.7413 | F1: 0.6923
Epoch 10 | Loss: 200.8949 | Val Acc: 0.7605 | F1: 0.7144
Epoch 11 | Loss: 189.7468 | Val Acc: 0.7702 | F1: 0.7246
Epoch 12 | Loss: 179.8772 | Val Acc: 0.7827 | F1: 0.7413
Epoch 13 | Loss: 170.5989 | Val Acc: 0.7812 | F1: 0.7392
Epoch 14 | Loss: 164.0885 | Val Acc: 0.7880 | F1: 0.7555
Epoch 15 | Loss: 158.4306 | Val Acc: 0.8041 | F1: 0.7613
Epoch 16 | Loss: 152.9763 | Val Acc: 0.8151 | F1: 0.7828
Epoch 17 | Loss: 147.4890 | Val Acc: 0.8116 | F1: 0.7767
Epoch 18 | Loss: 142.9594 | Val Acc: 0.8124 | F1: 0.7822
Epoch 19 | Loss: 137.8456 | Val Acc: 0.8296 | F1: 0.8001
Epoch 20 | Loss: 134.9501 | Val Acc: 0.8381 | F1: 0.8050
Epoch 21 | Loss: 129.8264 | Val Acc: 0.8289 | F1: 0.7983
Epoch 22 | Loss: 129.4190 | Val Acc: 0.8396 | F1: 0.8127
Epoch 23 | Loss: 125.7412 | Val Acc: 0.8360 | F1: 0.8058
Epoch 24 | Loss: 123.7982 | Val Acc: 0.8407 | F1: 0.8096
Epoch 25 | Loss: 120.7942 | Val Acc: 0.8444 | F1: 0.8152
Epoch 26 | Loss: 120.4952 | Val Acc: 0.8357 | F1: 0.7949
Epoch 27 | Loss: 117.6459 | Val Acc: 0.8442 | F1: 0.8162
Epoch 28 | Loss: 114.4057 | Val Acc: 0.8500 | F1: 0.8213
Epoch 29 | Loss: 115.1669 | Val Acc: 0.8490 | F1: 0.8222
Epoch 30 | Loss: 111.0075 | Val Acc: 0.8319 | F1: 0.7952
Epoch 31 | Loss: 110.7615 | Val Acc: 0.8477 | F1: 0.8186
Epoch 32 | Loss: 110.4251 | Val Acc: 0.8387 | F1: 0.8093
Epoch 33 | Loss: 106.4431 | Val Acc: 0.8452 | F1: 0.8145
Epoch 34 | Loss: 105.1925 | Val Acc: 0.8443 | F1: 0.8194
""",
    "L6_H512": """Epoch 01 | Loss: 515.3894 | Val Acc: 0.2850 | F1: 0.1519
Epoch 02 | Loss: 468.6768 | Val Acc: 0.3727 | F1: 0.2401
Epoch 03 | Loss: 415.7446 | Val Acc: 0.4671 | F1: 0.3750
Epoch 04 | Loss: 373.7504 | Val Acc: 0.5489 | F1: 0.4479
Epoch 05 | Loss: 335.0744 | Val Acc: 0.5939 | F1: 0.5135
Epoch 06 | Loss: 303.2132 | Val Acc: 0.6298 | F1: 0.5500
Epoch 07 | Loss: 274.3578 | Val Acc: 0.6654 | F1: 0.5991
Epoch 08 | Loss: 254.1141 | Val Acc: 0.6919 | F1: 0.6320
Epoch 09 | Loss: 235.3105 | Val Acc: 0.7160 | F1: 0.6642
Epoch 10 | Loss: 217.7647 | Val Acc: 0.7478 | F1: 0.6943
Epoch 11 | Loss: 203.2475 | Val Acc: 0.7532 | F1: 0.7013
Epoch 12 | Loss: 193.4812 | Val Acc: 0.7684 | F1: 0.7216
Epoch 13 | Loss: 182.7016 | Val Acc: 0.7677 | F1: 0.7258
Epoch 14 | Loss: 174.9117 | Val Acc: 0.7872 | F1: 0.7413
Epoch 15 | Loss: 168.1983 | Val Acc: 0.8055 | F1: 0.7671
Epoch 16 | Loss: 162.9324 | Val Acc: 0.7935 | F1: 0.7514
Epoch 17 | Loss: 156.0695 | Val Acc: 0.8080 | F1: 0.7721
Epoch 18 | Loss: 150.6555 | Val Acc: 0.8217 | F1: 0.7872
Epoch 19 | Loss: 148.2048 | Val Acc: 0.8217 | F1: 0.7868
Epoch 20 | Loss: 142.5402 | Val Acc: 0.8230 | F1: 0.7895
Epoch 21 | Loss: 138.9046 | Val Acc: 0.8334 | F1: 0.8051
Epoch 22 | Loss: 134.0381 | Val Acc: 0.8289 | F1: 0.7976
Epoch 23 | Loss: 131.7830 | Val Acc: 0.8297 | F1: 0.7990
Epoch 24 | Loss: 129.7110 | Val Acc: 0.8323 | F1: 0.7938
Epoch 25 | Loss: 127.2538 | Val Acc: 0.8420 | F1: 0.8116
Epoch 26 | Loss: 123.1012 | Val Acc: 0.8406 | F1: 0.8116
Epoch 27 | Loss: 122.0394 | Val Acc: 0.8427 | F1: 0.8108
Epoch 28 | Loss: 118.4085 | Val Acc: 0.8384 | F1: 0.8088
Epoch 29 | Loss: 119.2958 | Val Acc: 0.8416 | F1: 0.8093
Epoch 30 | Loss: 115.5702 | Val Acc: 0.8454 | F1: 0.8149
Epoch 31 | Loss: 112.7049 | Val Acc: 0.8470 | F1: 0.8188
Epoch 32 | Loss: 113.3342 | Val Acc: 0.8453 | F1: 0.8142
Epoch 33 | Loss: 110.4768 | Val Acc: 0.8473 | F1: 0.8177
Epoch 34 | Loss: 109.0242 | Val Acc: 0.8467 | F1: 0.8163
Epoch 35 | Loss: 108.1480 | Val Acc: 0.8542 | F1: 0.8213
Epoch 36 | Loss: 105.2037 | Val Acc: 0.8482 | F1: 0.8156
Epoch 37 | Loss: 104.5930 | Val Acc: 0.8516 | F1: 0.8146
Epoch 38 | Loss: 103.7672 | Val Acc: 0.8499 | F1: 0.8186
Epoch 39 | Loss: 100.9747 | Val Acc: 0.8476 | F1: 0.8174
Epoch 40 | Loss: 102.1221 | Val Acc: 0.8484 | F1: 0.8228
Epoch 41 | Loss: 100.6198 | Val Acc: 0.8609 | F1: 0.8331
Epoch 42 | Loss: 98.2104 | Val Acc: 0.8635 | F1: 0.8367
Epoch 43 | Loss: 99.1781 | Val Acc: 0.8606 | F1: 0.8352
Epoch 44 | Loss: 96.6221 | Val Acc: 0.8576 | F1: 0.8310
Epoch 45 | Loss: 96.6938 | Val Acc: 0.8632 | F1: 0.8337
Epoch 46 | Loss: 96.5833 | Val Acc: 0.8574 | F1: 0.8258
Epoch 47 | Loss: 95.4757 | Val Acc: 0.8574 | F1: 0.8325
""",
    "L6_H1024": """Epoch 01 | Loss: 531.5759 | Val Acc: 0.2572 | F1: 0.0815
Epoch 02 | Loss: 477.3217 | Val Acc: 0.3346 | F1: 0.1810
Epoch 03 | Loss: 442.5713 | Val Acc: 0.4278 | F1: 0.2991
Epoch 04 | Loss: 406.7996 | Val Acc: 0.4876 | F1: 0.3841
Epoch 05 | Loss: 373.5131 | Val Acc: 0.5583 | F1: 0.4807
Epoch 06 | Loss: 340.0600 | Val Acc: 0.5927 | F1: 0.5003
Epoch 07 | Loss: 313.5079 | Val Acc: 0.6315 | F1: 0.5432
Epoch 08 | Loss: 293.3234 | Val Acc: 0.6513 | F1: 0.5665
Epoch 09 | Loss: 273.8821 | Val Acc: 0.6839 | F1: 0.6162
Epoch 10 | Loss: 261.3257 | Val Acc: 0.6690 | F1: 0.6109
Epoch 11 | Loss: 245.8630 | Val Acc: 0.7093 | F1: 0.6488
Epoch 12 | Loss: 234.4745 | Val Acc: 0.7098 | F1: 0.6490
Epoch 13 | Loss: 222.2374 | Val Acc: 0.7331 | F1: 0.6720
Epoch 14 | Loss: 212.0219 | Val Acc: 0.7544 | F1: 0.6981
Epoch 15 | Loss: 203.0588 | Val Acc: 0.7581 | F1: 0.7025
Epoch 16 | Loss: 192.5781 | Val Acc: 0.7794 | F1: 0.7283
Epoch 17 | Loss: 188.3493 | Val Acc: 0.7684 | F1: 0.7153
Epoch 18 | Loss: 179.1020 | Val Acc: 0.7933 | F1: 0.7535
Epoch 19 | Loss: 175.0280 | Val Acc: 0.7904 | F1: 0.7457
Epoch 20 | Loss: 170.0580 | Val Acc: 0.8037 | F1: 0.7628
Epoch 21 | Loss: 164.0748 | Val Acc: 0.8017 | F1: 0.7609
Epoch 22 | Loss: 160.5622 | Val Acc: 0.8168 | F1: 0.7760
Epoch 23 | Loss: 154.5573 | Val Acc: 0.8100 | F1: 0.7704
Epoch 24 | Loss: 151.0926 | Val Acc: 0.8064 | F1: 0.7708
Epoch 25 | Loss: 146.9401 | Val Acc: 0.8178 | F1: 0.7785
Epoch 26 | Loss: 146.0370 | Val Acc: 0.8248 | F1: 0.7882
Epoch 27 | Loss: 143.1744 | Val Acc: 0.8259 | F1: 0.7924
Epoch 28 | Loss: 141.2738 | Val Acc: 0.8351 | F1: 0.7986
Epoch 29 | Loss: 136.5064 | Val Acc: 0.8336 | F1: 0.8013
Epoch 30 | Loss: 134.9306 | Val Acc: 0.8360 | F1: 0.8023
Epoch 31 | Loss: 131.5147 | Val Acc: 0.8350 | F1: 0.8021
Epoch 32 | Loss: 129.6913 | Val Acc: 0.8351 | F1: 0.8031
Epoch 33 | Loss: 127.1407 | Val Acc: 0.8339 | F1: 0.8011
Epoch 34 | Loss: 127.5120 | Val Acc: 0.8306 | F1: 0.7964
Epoch 35 | Loss: 126.0851 | Val Acc: 0.8367 | F1: 0.8038
Epoch 36 | Loss: 124.1777 | Val Acc: 0.8424 | F1: 0.8089
Epoch 37 | Loss: 121.4695 | Val Acc: 0.8432 | F1: 0.8098
Epoch 38 | Loss: 120.1578 | Val Acc: 0.8270 | F1: 0.7966
Epoch 39 | Loss: 116.7172 | Val Acc: 0.8467 | F1: 0.8165
Epoch 40 | Loss: 116.2481 | Val Acc: 0.8452 | F1: 0.8138
Epoch 41 | Loss: 114.6788 | Val Acc: 0.8432 | F1: 0.8096
Epoch 42 | Loss: 115.7447 | Val Acc: 0.8462 | F1: 0.8165
Epoch 43 | Loss: 111.7040 | Val Acc: 0.8467 | F1: 0.8183
Epoch 44 | Loss: 111.4711 | Val Acc: 0.8502 | F1: 0.8200
Epoch 45 | Loss: 109.7338 | Val Acc: 0.8469 | F1: 0.8064
Epoch 46 | Loss: 108.2416 | Val Acc: 0.8483 | F1: 0.8204
Epoch 47 | Loss: 107.5188 | Val Acc: 0.8567 | F1: 0.8305
Epoch 48 | Loss: 109.4334 | Val Acc: 0.8529 | F1: 0.8244
Epoch 49 | Loss: 106.8006 | Val Acc: 0.8573 | F1: 0.8299
Epoch 50 | Loss: 103.5964 | Val Acc: 0.8566 | F1: 0.8284
Epoch 51 | Loss: 104.5127 | Val Acc: 0.8514 | F1: 0.8205
Epoch 52 | Loss: 104.2665 | Val Acc: 0.8583 | F1: 0.8254
"""
}

# Parse all configurations
all_data = pd.concat([parse_metrics(data, name) for name, data in configurations.items()], ignore_index=True)

# Plot the six required graphs
metrics = ['Loss', 'Accuracy', 'F1 Score']
for var in ['Layers', 'Hidden Channels']:
    for metric in metrics:
        plot_metric_impact(all_data, var, metric)
plt.tight_layout()
plt.show()