from types import SimpleNamespace

config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
    "hidden_size": 72,
    "num_layers": 5,
    "output_size": 3,
    "dropout": 0.2,
    # "input_size": 10,  # 입력 특성의 수
    # "sequence_length": 30,  # 시계열 데이터의 길이
    # "optimizer": "adamw",

    "loss_function": "mae",
    # "scheduler": "reduce_lr_on_plateau",
    # Early stopping 설정
    "early_stopping": True,
    "patience": 10,
    "min_delta": 0.001
}

CFG = SimpleNamespace(**config)

품목_리스트 = ['건고추', '사과', '감자', '배', '깐마늘(국산)', '무', '상추', '배추', '양파', '대파']