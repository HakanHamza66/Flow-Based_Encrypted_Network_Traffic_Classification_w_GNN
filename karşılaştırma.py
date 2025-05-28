import matplotlib.pyplot as plt
import re

data = """
"L5_H512": Epoch 01 | Loss: 507.1274 | Val Acc: 0.2927 | F1: 0.1414
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
"""

# Extracting data using regex
epochs = []
val_accs = []
f1_scores = []

# Improved regex to handle the first line with "L5_H512"
for line in data.strip().split('\n'):
    match = re.search(r"Epoch (\d+) \|.*?Val Acc: (\d\.\d+) \| F1: (\d\.\d+)", line)
    if match:
        epochs.append(int(match.group(1)))
        val_accs.append(float(match.group(2)))
        f1_scores.append(float(match.group(3)))

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(epochs, val_accs,  linestyle='-', label='Validation Accuracy (Val Acc)')
plt.plot(epochs, f1_scores,  linestyle='-', label='F1 Score')

# Adding the horizontal SVM line
svm_threshold = 0.4084
plt.axhline(y=svm_threshold, color='r', linestyle=':', linewidth=2, label=f'SVM F1 Score ({svm_threshold})')
svm_threshold = 0.4934
plt.axhline(y=svm_threshold, color='g', linestyle=':', linewidth=2, label=f'SVM Accuracy Value ({svm_threshold})')

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Validation Accuracy and F1 Score vs. Epoch')
plt.legend()
plt.grid(True)
plt.xticks(epochs[::5]) # Show every 5th epoch on x-axis for clarity
plt.yticks([i/10 for i in range(0, 11)]) # y-axis ticks from 0.0 to 1.0
plt.ylim(0, 1.0) # Set y-axis limits

# Show plot
plt.tight_layout()
plt.show()