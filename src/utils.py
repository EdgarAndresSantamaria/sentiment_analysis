from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
id2label = {1: "--", 2: "-", 3: "0", 4: "+", 5: "++"}
label2id = {"--": 1, "-": 2, "0": 3, "+": 4, "++": 5}

def load_model(model_name="bert-base-cased"):
    """
    Loads a trained model and tokenizer
    """

    model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(id2label.keys()),
            id2label=id2label,
            label2id=label2id,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
