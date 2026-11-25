import os
from ucimlrepo import fetch_ucirepo

concrete = fetch_ucirepo(id=165)           # 165 = id датасета Concrete на UCI
X = concrete.data.features                 # DataFrame: 8 входов
y = concrete.data.targets                  # DataFrame: 1 колонка (strength)

print(concrete.metadata)                   # источник, размер, пропуски...
print(concrete.variables)                  # имя, роль, тип, единицы каждой переменной

old_names = list(X.columns)                # запоминаем оригинальные названия

X = X.copy()
X.columns = ["cement", "slag", "ash", "water", "superplastic",
             "coarse_agg", "fine_agg", "age"]
y = y.iloc[:, 0].rename("strength")        # DataFrame -> Series

# проверка: оригинальное имя <-> новое имя, порядок должен совпасть
for old, new in zip(old_names, X.columns):
    print(f"{old:35s} -> {new}")

os.makedirs("data", exist_ok=True)
X.assign(strength=y).to_csv("data/concrete.csv", index=False)