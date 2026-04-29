import torch
import numpy as np
import os
import pandas as pd


# Проверяем доступна ли GPU и задаем вычислительное устройство
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('Available GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU')
    device = torch.device("cpu")

# Загрузка данных реализована на основе pandas dataframe
df = pd.read_csv("./data2/vocabalv7.csv", delimiter='.', header=0, names=['sentence', 'label'])

print('В наборе предложений: {:,}\n'.format(df.shape[0]))

# Пример
df.sample(10)

# Нас интересуют метки классов и сами предложения, на них мы будем обучать нашу сеть
sentences = df.sentence.values
labels = df.label.values

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

# Токенизатор — стандартный предобученный
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

##### размер нового словаря
tokenizer.vocab_size

sl = [len(tokenizer.encode(sen, add_special_tokens=True)) for sen in sentences]
print('Максимальная длина предложения:', max(sl))
print('Минимальная длина предложения:', min(sl))

# Используем реальный максимум (чтобы ничего не обрезать)
max_len = max(sl)
print(f'Будем использовать max_len = {max_len} (все предложения поместятся)')


input_ids = np.zeros((len(sentences), max_len), dtype=int)

# Каждое предложение энкодится по отдельности — БЕЗ truncation
for s, i in zip(sentences, range(len(sentences))):
    enc_s = tokenizer.encode(s,                      
                        add_special_tokens=True,  # служебные токены [CLS], [SEP]
                        padding='max_length',     # дополняем до max_len нулями
                        max_length=max_len        # реальный максимум
                   )
    input_ids[i,] = enc_s


# Создаем attention mask для виртуальных токенов
attention_masks = []

for s in input_ids:
    # Если ID = 0, это виртуальный токен (падинг) и маска для него 0.
    # Если ID > 0, это реальный токен и маска для него 1.
    att_mask = [int(id_ > 0) for id_ in s]
    attention_masks.append(att_mask)



# Формируем тестовый и валидационный набор
from sklearn.model_selection import train_test_split

# разбиваем данные, метки классов
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, test_size=0.1)
# и маску
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, test_size=0.1)

# все конвертируем в тензоры
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# теперь можно создавать Dataset и DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# размер батча придется уменьшить (если на GPU), т.к. за счет
# увеличившегося словаря выросла и модель 
batch_size = 4

# Пакуем в тренировочный предложения (ID), маску и метки классов
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# и в валидационный
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


# теперь можно переходить к заданию модели

from transformers import BertForSequenceClassification


model = BertForSequenceClassification.from_pretrained(
    'DeepPavlov/rubert-base-cased',  # имя модели
    num_labels=3  
)

# Отправляем модель на GPU
if torch.cuda.is_available():
    model.cuda()
    
optimizer = torch.optim.AdamW(model.parameters(),
                  lr=2e-5,  # скорость обучения
                  eps=1e-8  # специфический параметр, повышающий стабильность обучения
                )

from transformers import get_linear_schedule_with_warmup

# Количество эпох обучения
epochs = 2

# Шагов обучения = number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# scheduler - планировщик изменяющий скорость обучения
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

# функция вычисления точности обучения
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


import time
import datetime
import random

# Задаем seed
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Тут храним наши лоссы
loss_values = []

# Цикл обучения будет состоять из обучения и валидации
for epoch_i in range(0, epochs):
    
    ################ Часть обучения #####################
    
    print("")
    print('Эпоха {:} из {:} '.format(epoch_i + 1, epochs))
    
    t0 = time.time()

    # потери за эпоху
    total_loss = 0
    
    # переключаем в режим обучения
    model.train()

    # пробегаем по батчам
    for step, batch in enumerate(train_dataloader):

       
        # Достаем из батча данные: предложения, маску и метки
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # обнуляем градиенты
        model.zero_grad()        

        # прямой проход
        outputs = model(b_input_ids.to(torch.long), 
                    token_type_ids=None, 
                    attention_mask=b_input_mask.to(torch.long), 
                    labels=b_labels.to(torch.long))
       
               # потери
        loss = outputs.loss
        total_loss += loss.item()

        # обратный проход
        loss.backward()

        # обрезаем градиенты до 1.0, чтобы предотвратить "взрыв градиентов"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # обновляем параметры модели
        optimizer.step()

        # изменяем скорость обучения
        scheduler.step()
        
         # диагностическую информацию выводим каждые 100 батчей
        if step % 10 == 0 and not step == 0:
            time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
            print(' Батч {:>4,} из {:>4,}. Затраченное время: {:}. Ошибка: {:}.'.format(step, len(train_dataloader), time_elapsed, loss))


    # средний loss 
    avg_train_loss = total_loss / len(train_dataloader)            
    
    # сохраним для графика
    loss_values.append(avg_train_loss)

    print("")
    print(" Средний loss: {0:.2f}".format(avg_train_loss))
    print(" Обучение эпохи прошло за: {:}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))))
        
    ################ Часть валидации #####################
    # Позволяет понять правильно ли мы учимся и учимся ли вообще

    print("\n Validation...")
    t0 = time.time()

    # модель в  evaluation режим
    model.eval()

    # диагностические переменные
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # пробегаем валидационный набор
    for batch in validation_dataloader:
        
        batch = tuple(t.to(device) for t in batch)
        
        # берем нужные данные из батча
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
            # прямой проход
            outputs = model(b_input_ids.to(torch.long), 
                    token_type_ids=None, 
                    attention_mask=b_input_mask.to(torch.long), 
                    labels=b_labels.to(torch.long))
        
        # "logits" хранят вероятности классов похоже на softmax
        logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # считаем точность модели на валидации
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        # суммарная точность
        eval_accuracy += tmp_eval_accuracy

        # сколько батчей прошло
        nb_eval_steps += 1

    # результат валидации
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Валидация прошла за: {:}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))))



# Можно построить график обучения 
import matplotlib.pyplot as plt

plt.plot(loss_values, 'b-o')
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


sentence = "ваш код не учитывает некоторые особенности языка"

# Токенизация (так же, как при обучении)
enc_s = tokenizer.encode(
    sentence,
    add_special_tokens=True,
    padding='max_length',
    max_length=max_len,
    truncation=True
)

# Формируем input_ids и attention_mask
input_ids = torch.tensor([enc_s]).to(device)
attention_mask = torch.tensor([[int(id_ > 0) for id_ in enc_s]]).to(device)

# Предсказание
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids.to(torch.long),
        attention_mask=attention_mask.to(torch.long),
        token_type_ids=None
    )

# Получаем результат
logits = outputs.logits
predicted_label = torch.argmax(logits, dim=1).item()

# Вывод
label_map = {0: "повествовательное", 1: "побудительное", 2: "критика"}
print(f"Предложение: \"{sentence}\"")
print(f"Тип: {label_map[predicted_label]} (метка: {predicted_label})")



