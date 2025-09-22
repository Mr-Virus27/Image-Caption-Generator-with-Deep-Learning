# Step 12: Evaluation
def evaluate_model():
    features = load_features()
    mapping = load_captions()
    clean_mapping(mapping)
    tokenizer = create_tokenizer(mapping)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for key in mapping for caption in mapping[key])
    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.90)
    train_ids, test_ids = image_ids[:split], image_ids[split:]
    model = ... # load trained model here
    actual, predicted = [], []
    for key in tqdm(test_ids):
        y_pred = predict_caption(model, features[key], tokenizer, max_length)
        actual_captions = [caption.split() for caption in mapping[key]]
        y_pred = y_pred.split()
        actual.append(actual_captions)
        predicted.append(y_pred)
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
