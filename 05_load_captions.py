# Step 5: Load Captions & Mapping
def load_captions():
    with open(os.path.join(BASEDIR, 'captions.txt'), 'r') as f:
        captions_doc = f.read()
    mapping = dict()
    for line in tqdm(captions_doc.split('
')):
        tokens = line.split(',')
        if len(tokens) < 2:
            continue
        image_id, caption = tokens[0].split('.')[0], tokens[1]
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping
