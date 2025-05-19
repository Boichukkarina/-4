import joblib

model = joblib.load('xgb_model_with_extra_flag.pkl')
tfidf = joblib.load('tfidf_vectorizer_with_extra_flag.pkl')
all_tags = [
    'Нравится скорость отработки заявок',
    'Нравится качество выполнения заявки',
    'Нравится качество работы сотрудников',
    'Понравилось выполнение заявки',
    'Вопрос решен',
    'Вопрос не решен'
]

def predict_labels(text):
    X = tfidf.transform([text])
    preds = model.predict(X)
    print(preds)
    preds = preds[0]
    
    # Загрузим список имен меток, чтобы их показать
    labels = all_tags
    
    # Формируем результат
    result = {label: bool(pred) for label, pred in zip(labels, preds)}
    return result

if __name__ == "__main__":
    while True:
        text = input("Введите сообщение для классификации (или 'exit' для выхода):\n")
        if text.lower() == 'exit':
            break
        prediction = predict_labels(text)
        print("Категории, выставленные моделью:")
        for cat, val in prediction.items():
            if val:
                print(f" - {cat}")