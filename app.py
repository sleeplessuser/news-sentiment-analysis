from functools import partial
from argparse import ArgumentParser

import gradio as gr

from transformers import pipeline

label_mappings = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("ckpt")
    return parser.parse_args()


def predict(model, txt):
    x = model(txt)[0]
    label = x["label"]
    return label_mappings[label]


if __name__ == "__main__":
    args = parse_args()
    model = pipeline("text-classification", args.ckpt)
    app = gr.Interface(
        fn=partial(predict, model),
        inputs="text",
        outputs=gr.Label(),
        allow_flagging=False,
    )
    app.launch()
