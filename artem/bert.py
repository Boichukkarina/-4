from transformers import BertForSequenceClassification, BertTokenizer
import torch
import pymorphy3
import re

# Загрузка модели BERT
model = BertForSequenceClassification.from_pretrained('bert_model_with_extra_flag')
tokenizer = BertTokenizer.from_pretrained('bert_token_model_with_extra_flag')

# Установка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

all_tags = [
    'Нравится скорость отработки заявок',
    'Нравится качество выполнения заявки',
    'Нравится качество работы сотрудников',
    'Понравилось выполнение заявки',
    'Вопрос решен',
    'Вопрос не решен'
]

def predict_labels(text):
    # Токенизация
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Предсказание
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Получение вероятностей
    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    
    # Классификация
    threshold = 0.5
    preds = (probs > threshold).astype(int)
    
    # Формирование результата
    result = {label: bool(pred) for label, pred in zip(all_tags, preds)}
    return result

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^а-яА-ЯёЁ0-9\s]', ' ', text)

    morph = pymorphy3.MorphAnalyzer()
    words = text.split()
    cleaned_words = []
    for word in words:
        parsed_word = morph.parse(word)[0]
        cleaned_words.append(parsed_word.normal_form)
    return " ".join(cleaned_words)

if __name__ == "__main__":
    while True:
        text = input('Введите сообщение для классификации ("выход" - прекратить ввод):\n')
        if text.lower() == 'выход':
            break
        prediction = predict_labels(clean_text(text))
        print(clean_text(text))
        print("Категории, выставленные моделью:")
        for cat, val in prediction.items():
            if val:
                print(f" - {cat}")
