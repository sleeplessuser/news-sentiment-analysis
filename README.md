# news-sentiment-analysis
## Project Structure
```bash
├── app.py # gradio web-app
├── pybooks 
│   ├── baseline.ipynb # tf-idf + random forest classifier baseline evaluation
│   ├── eda.ipynb # dataset statistics
│   └── eval.ipynb # trained model evaluation
├── requirements.txt # python-dependencies
├── train.py # script for model training
└── utils.py # utility functions
```
## Installation
```bash
git clone https://github.com/sleeplessuser/news-sentiment-analysis
cd news-sentiment-analysis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training
```bash
train.py [--lr LR] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--weight-decay WEIGHT_DECAY] [--num-classes NUM_LABELS] [--model-id MODEL_ID]
```

## Inference with web-ui
```bash
python app.py <ckpt_directory>
```
