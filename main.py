from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

app = Flask(__name__)

# 모델 및 스케일러 로드
model = tf.keras.models.load_model('kia_large_model.h5')
scaler = joblib.load('kia_large_scaler.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_features = np.array(data['features']).reshape(1, -1)

        # 데이터 스케일링하기
        input_scaled = scaler.transform(input_features)
        input_scaled = np.reshape(input_scaled, (input_scaled.shape[0], input_scaled.shape[1], 1))

        # 예측하기
        prediction = model.predict(input_scaled)

        # 스케일링 복원
        predicted_price = scaler.inverse_transform(
            np.hstack([prediction, np.zeros((prediction.shape[0], scaler.n_features_in_ - 1))])
        )[:, 0]

        return jsonify({'predicted_price': float(predicted_price[0])})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)