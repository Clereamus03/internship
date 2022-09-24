import pandas as pd

file_name = input("Enter file name ")
df = pd.read_csv(file_name)
#print(df)
columns_list = list(df.columns)
print("From the following columns select the dependent feature: \n", columns_list)
dep_feat = input("Enter the dependent feature: ")
print(len(set(df[dep_feat])), len(df[dep_feat]))
uniq = len(set(df[dep_feat]))
tot = len(df[dep_feat])
if (uniq/tot)*100 < 5:
    print("Classification")
else:
    print("Regression")
print((uniq/tot)*100)