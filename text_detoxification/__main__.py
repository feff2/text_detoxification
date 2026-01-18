"""Entry point for running as module."""

import fire
from text_detoxification.cli import convert, infer, serve, train

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "convert": convert,
            "serve": serve,
        }
    )

