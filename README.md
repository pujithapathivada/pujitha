# House Price Prediction

A simple ML project that predicts house prices in India using a Random Forest model. There's also a small Streamlit web app to try predictions interactively.

## Files

- `ml.ipynb` — main notebook with data cleaning, EDA, model training and evaluation
- `house_prices.csv` — the dataset
- `train_and_save.py` — script to train the model and save it
- `app.py` — Streamlit app for predictions
- `requirements.txt` — Python packages needed

## How to run

1. Install the packages:

```
pip install -r requirements.txt
```

2. Train the model (this saves `rf_model.pkl` and `encoders.pkl`):

```
python train_and_save.py
```

3. Start the app:

```
streamlit run app.py
```

Then just enter house details in the browser and click Predict.