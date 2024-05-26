class Config:
    # data
    WINDOW_SIZE = 5
    FEATURE_SIZE = 6
    INPUT_SIZE = WINDOW_SIZE * 6
    DATASET = 'Mouse Data.csv'

    # training
    BATCH_SIZE = 64
    TRAIN_SIZE = 0.8
    LEARNING_RATE = 0.1
    EPOCHS = 350
    EARLY_STOPPING_PATIENCE = 20
    WEIGHT_DECAY = 1e-5
    SCHEDULER_PATIENCE = 10

    # backend server
    HOST = 'localhost'
    PORT = 8765
