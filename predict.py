import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

def generate_predictions(model, test_data,test_loader, scaler, price_column):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch)
            
            predictions.append(output.numpy())
    
    predictions_array = np.concatenate(predictions)
    
    # 예측값을 원래 스케일로 복원
    price_column_index = test_data.columns.get_loc(price_column)
    predictions_reshaped = predictions_array.reshape(-1, 1)
    
    # 가격 열에 대해서만 inverse_transform 적용
    price_scaler = MinMaxScaler()
    price_scaler.min_ = scaler.min_[price_column_index]
    price_scaler.scale_ = scaler.scale_[price_column_index]
    predictions_original_scale = price_scaler.inverse_transform(predictions_reshaped)
    
    return predictions_original_scale.flatten()

def prepare_submission(품목별_predictions, file_name):
    sample_submission = pd.read_csv('./sample_submission.csv')
    for 품목명, predictions in 품목별_predictions.items():
        if len(predictions) != len(sample_submission):
            print(f"경고: {품목명}의 예측 수 ({len(predictions)})가 샘플 제출 파일의 행 수 ({len(sample_submission)})와 일치하지 않습니다.")
        sample_submission[품목명] = predictions
    sample_submission.to_csv(f'./{file_name}.csv', index=False)