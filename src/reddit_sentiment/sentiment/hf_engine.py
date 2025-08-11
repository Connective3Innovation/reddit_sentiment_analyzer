from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
class HfEngine:
    def __init__(self):
        model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Hard-cap to 512 to avoid position-id overflow
        self.tokenizer.model_max_length = 512

        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        device = 0 if torch.cuda.is_available() else -1

        self._pipe = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )

    def run(self, texts):
        # Defensive: ensure list[str] and drop weird values
        texts = [t if isinstance(t, str) else "" for t in texts]

        # Batch with explicit truncation/max_length
        outputs = self._pipe(
            texts,
            batch_size=32,
            truncation=True,
            max_length=512,   # IMPORTANT: 512, not 514
        )
        df = pd.DataFrame(outputs)
        return df.rename(columns={"label": "sentiment", "score": "prob"})
