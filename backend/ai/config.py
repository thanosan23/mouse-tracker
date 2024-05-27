class Config:
    # data
    WINDOW_SIZE = 10
    FEATURE_SIZE = 5
    INPUT_SIZE = WINDOW_SIZE * FEATURE_SIZE
    DATASET = 'Mouse Data.csv'

    # training
    BATCH_SIZE = 128
    TRAIN_SIZE = 0.8
    LEARNING_RATE = 0.1
    EPOCHS = 350
    EARLY_STOPPING_PATIENCE = 50
    WEIGHT_DECAY = 1e-5
    SCHEDULER_PATIENCE = 10

    # backend server
    HOST = 'localhost'
    PORT = 8765
