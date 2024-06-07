# Mouse Tracker 

This project explores how non-deterministic assumptions can control interaction.

This project specifically sees how mouse prediction can improve interaction.


## How to use the mouse tracker 
### Training the model

#### Obtaining the dataset
To create and train your own dataset, go to `mouse_tracker/src/pages/index.tsx` and set the `TRAIN` variable to `true`. Then, you can run `npm run dev` in the `mouse_tracker` folder. You can then click the buttons continously to track your mouse and when you are done, click "Export CSV" to save the dataset. You can then bring this dataset into `backend/ai/` folder.

#### Running the training script
To train the model, run: 
```
cd backend/ai
python3 train.py
```

You have now officially trained the model.

## Testing out your model
Go to `mouse_tracker/src/pages/index.tsx` and set `TRAIN` to `false`. You can then run `npm run dev` to run the page that predicts where your mouse will go.

Make sure to also run the backend by running:
```
cd backend
python3 backend.py
```
