# Step 6: Clean Captions
def clean_mapping(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].lower()
            caption = ''.join([c for c in caption if c.isalpha() or c.isspace()])
            caption = ' '.join([word for word in caption.split() if len(word) > 1])
            captions[i] = 'startseq ' + caption + ' endseq'
