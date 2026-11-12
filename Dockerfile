FROM tensorflow/tensorflow:2.18.0-gpu

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "forex_prediction_model.trading_api:app", "--host", "0.0.0.0", "--port", "8000"]
