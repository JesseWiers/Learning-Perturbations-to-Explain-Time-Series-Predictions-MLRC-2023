try:
    from transformers.models.bert import (
        BertConfig,
        BertTokenizer,
        BertForSequenceClassification,
    )
except ImportError:
    BertConfig = None
    BertForSequenceClassification = None


def Bert(pretrained_model_name_or_path: str = None, config=None, **kwargs):
    """
    Get Bert model for sentence classification, either as a pre-trained model
    or from scratch. Any transformers model could theoretically be used, this
    method is provided as an example.

    Args:
        pretrained_model_name_or_path: Path of the pre-trained model.
            If ``None``, return an untrained Bert model. Default to ``None``
        config: Config of the Bert. Required when not loading a pre-trained
            model, otherwise unused. Default to ``None``
        kwargs: Additional arguments for the tokenizer if not pretrained.

    Returns:
        BertForSequenceClassification: Bert model for sentence classification.

    References:
        https://huggingface.co/docs/transformers/main/en/model_doc/bert
    """
    assert BertConfig is not None, "transformers is not installed."

    if pretrained_model_name_or_path is None:
        assert config is not None, "Bert config must be provided."
        return (
            BertTokenizer(config.vocab_size),
            BertForSequenceClassification(config=config),
        )

    return (
        BertTokenizer.from_pretrained(pretrained_model_name_or_path),
        BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            return_dict=False,
        ),
    )
