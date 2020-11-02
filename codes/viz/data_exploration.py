import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_json("../train.jsonl", lines=True)
print(df_test.head())


fig, ax = plt.subplots()
df_train_raw['Freq'].hist(ax=ax, bins=10, bottom=0.1)
ax.set_yscale('log')
plt.show()
df_train_raw['label'].hist()
plt.show()

