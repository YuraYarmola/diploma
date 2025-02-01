# Завантаження класів з файлу
with open("dataset/classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Формування `data.yaml`
data_yaml = f"""train: dataset/images/train
val: dataset/images/train

nc: {len(class_names)}
names: {class_names}
"""

# Збереження у файл
with open("dataset/data.yaml", "w") as f:
    f.write(data_yaml)

print("Файл data.yaml збережено!")
