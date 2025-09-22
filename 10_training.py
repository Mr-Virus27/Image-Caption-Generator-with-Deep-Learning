# Step 10: Model Training Loop (Example)
def train_model():
    features = load_features()
    mapping = load_captions()
    clean_mapping(mapping)
    tokenizer = create_tokenizer(mapping)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for key in mapping for caption in mapping[key])
    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.90)
    train_ids, test_ids = image_ids[:split], image_ids[split:]
    model = define_model(vocab_size, max_length)
    batch_size = 32
    steps = len(train_ids) // batch_size
    generator = data_generator(train_ids, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    epochs = 10
    for i in range(epochs):
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save(os.path.join(WORKINGDIR, 'trainedmodel.h5'))
