import dill

import pandas as pd
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open('data/sber_auto_subscribing.pkl', 'rb') as f:
    model = dill.load(f)

# поле id я добавил только для связки исходных данных с предсказанием для них, никакой другой смысловой нагрузки оно не несет
class Form(BaseModel):
    id: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    geo_country: str

class Prediction(BaseModel):
    id: int
    pred: int

@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.model_dump()]).drop("id", axis=1)
    # известные значения входного датафрейма (одиночной строки) преобразуем в пригодные для модели
    # Получаем первую строку как Series
    row = df.iloc[0]
    # Формируем новые имена колонок
    new_columns = [f"{col}_{row[col]}" for col in df.columns]
    # Создаем новый DataFrame из одной строки, где значения — единицы
    df_new = pd.DataFrame([1] * len(new_columns), index=new_columns).T
    # Обнуляем индексы
    df_new.reset_index(drop=True, inplace=True)
    # теперь нам нужно остальные значения, которые требует модель, заполнить нулями, причем набор значений очень большой, и мы его получим отдельно из самой модели
    feature_list = model['model'].feature_names_in_
    # Приводим DataFrame к нужному виду, добавляя отсутствующие признаки и заполняя их нулями
    df_full = df_new.reindex(columns=feature_list, fill_value=0)
    y = model['model'].predict(df_full)
    result = {
        'id': form.id,
        'pred': y[0],
    }
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
