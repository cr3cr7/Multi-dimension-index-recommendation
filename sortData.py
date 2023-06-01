import pandas as pd

order_col = "Color"
df = pd.read_csv("~/multi-dimension/datasets/dmv-tiny.csv")
df.sort_values(by=order_col, inplace=True)
df.to_csv("~/multi-dimension/datasets/dmv-tiny-sort.csv",index=False)