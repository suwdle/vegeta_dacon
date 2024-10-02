from config import CFG, 품목_리스트

from data_processing import process_data
from dataset import AgriculturePriceDataset
from model import PricePredictionLSTM
from train import train_model, evaluate_model
from predict import generate_predictions, prepare_submission
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

def main():
    품목별_predictions = {}
    품목별_scalers = {}

    pbar = tqdm(품목_리스트, desc="품목 처리 중")
    for 품목명 in pbar:
        pbar.set_description(f"품목 처리 중: {품목명}")

        # 데이터 로드 및 전처리
        train_data, scaler = process_data("./train/train.csv", 
                                          "./train/meta/TRAIN_산지공판장_2018-2021.csv", 
                                          "./train/meta/TRAIN_전국도매_2018-2021.csv", 
                                          품목명)
        품목별_scalers[품목명] = scaler

        # 데이터셋 생성
        dataset = AgriculturePriceDataset(train_data)
        train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
        
        train_loader = DataLoader(train_data, batch_size=CFG.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=CFG.batch_size, shuffle=False)

        # 모델 초기화
        input_size = len(dataset.numeric_columns)
        model = PricePredictionLSTM(input_size, CFG.hidden_size, CFG.num_layers, CFG.output_size)
        criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate)

        # 모델 훈련
        best_val_loss = float('inf')
        patience = CFG.patience
        counter = 0
        for epoch in range(CFG.epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer)
            val_loss = evaluate_model(model, val_loader, criterion)
            
            if val_loss < best_val_loss - CFG.min_delta:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'models/best_model_{품목명}.pth')
                counter = 0
            else:
                counter += 1
            
            print(f'Epoch {epoch+1}/{CFG.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if CFG.early_stopping and counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

        print(f'Best Validation Loss for {품목명}: {best_val_loss:.4f}')

        # 테스트 데이터에 대한 예측 생성
        품목_predictions = []
        pbar_inner = tqdm(range(25), desc="테스트 파일 추론 중", position=1, leave=False)
        for i in pbar_inner:
            test_file = f"./test/TEST_{i:02d}.csv"
            산지공판장_file = f"./test/meta/TEST_산지공판장_{i:02d}.csv"
            전국도매_file = f"./test/meta/TEST_전국도매_{i:02d}.csv"
            
            test_data, _ = process_data(test_file, 산지공판장_file, 전국도매_file, 품목명, scaler=품목별_scalers[품목명])
            test_dataset = AgriculturePriceDataset(test_data, is_test=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            predictions = generate_predictions(model, test_data, test_loader, 품목별_scalers[품목명], dataset.price_column)
            품목_predictions.extend(predictions)

        품목별_predictions[품목명] = 품목_predictions

    # 제출 파일 준비
    file_name = 'early_stopping_submission'
    prepare_submission(품목별_predictions, file_name)

if __name__ == "__main__":
    main()