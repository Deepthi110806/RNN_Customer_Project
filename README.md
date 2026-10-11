# RNN-Based Text Classification of Customer Messages

A Bidirectional LSTM (a Recurrent Neural Network) that reads short customer messages
word by word and classifies them into one of **77 intent categories**
(e.g. `card_arrival`, `change_pin`, `declined_transfer`). The trained model is then used
to predict categories for previously unseen messages.

## Results
| Metric | Value |
|---|---|
| Test accuracy (3,080 unseen messages) | **86.0 %** |
| Macro F1 | **0.861** |
| Classes | 77 |

Sample predictions on new messages (see `screenshots/output.png`, `results/unseen_predictions.txt`):

| Message | Predicted |
|---|---|
| My new card still has not shown up in the mail | card_arrival (92%) |
| How do I change my PIN number? | change_pin (100%) |
| I was charged twice for the same purchase | transaction_charged_twice (100%) |
| Why was my transfer declined? | declined_transfer (99.8%) |

Ambiguous messages (e.g. "exchange rate if I pay abroad") get split confidence across
related categories, which is where most of the model's errors come from.

## Project structure
```
customer-rnn-project/
├── README.md
├── requirements.txt
├── dataset/            customer_train.csv, customer_test.csv
├── src/                rnn_text_classification.py   (train + evaluate + predict)
├── notebooks/          RNN_Analysis.ipynb           (step-by-step walkthrough)
├── results/            training_curves.png, per_class_f1.png, metrics.json,
│                       classification_report.txt, unseen_predictions.txt,
│                       training_log.txt, rnn_classifier.keras, labels.json
└── screenshots/        output.png
```

## How it works
1. **Data** – 10,003 training and 3,080 test messages, each with a category label.
   10 % of training data is held out for validation; the test set is only used at the end.
2. **Text -> sequence** – `TextVectorization` lowercases, strips punctuation and maps the
   words to integer IDs (vocabulary 8,000, max length 40).
3. **Model** – `Embedding(128)` -> `Bidirectional(LSTM(128))` -> `Dropout` -> `Dense(128)` -> `Dropout` -> `Dense(77, softmax)`.
   The LSTM reads the sequence in both directions so each word is understood in context.
4. **Training** – Adam, sparse categorical cross-entropy, early stopping on validation accuracy.
5. **Prediction** – the vectorizer is built into the saved model, so it accepts raw text.

## Run it
```bash
pip install -r requirements.txt
python src/rnn_text_classification.py                      # train + evaluate + demo
python src/rnn_text_classification.py --predict "my card has not arrived"
jupyter notebook notebooks/RNN_Analysis.ipynb
```
Training takes roughly 10 minutes on a CPU.

## Dataset
BANKING77 (Casanueva et al., 2020, PolyAI): 13,083 online-banking customer queries labelled
with 77 intents. Source used here:
https://github.com/PolyAI-LDN/task-specific-datasets (`banking_data/`). The same dataset is also
mirrored on Kaggle.

## Acknowledgements
- Dataset: PolyAI BANKING77 (above).
- Libraries: TensorFlow/Keras, scikit-learn, pandas, matplotlib.
- Code was developed with AI assistance (Claude); add your own notes here on what you changed or extended.
