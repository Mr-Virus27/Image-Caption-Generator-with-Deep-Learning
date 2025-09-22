# Step 7: Prepare Tokenizer
def create_tokenizer(mapping):
    all_captions = []
    for key in mapping:
        all_captions.extend(mapping[key])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer
