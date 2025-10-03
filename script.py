import os
import zipfile

project_dir = "image_caption_steps"
os.makedirs(project_dir, exist_ok=True)

# Step code mapping
sections = {
    "01_imports.py": '''# Step 1: Import Libraries
import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from PIL import Image
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
''',
    "02_paths.py": '''# Step 2: Set Paths
BASEDIR = 'datasets'
WORKINGDIR = 'imgcap_work'
''',
    "03_feature_extraction.py": '''# Step 3: Feature Extraction
def extract_features(directory):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = dict()
    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = img_name.split('.')[0]
        features[image_id] = feature
    pickle.dump(features, open(os.path.join(WORKINGDIR, 'features.pkl'), 'wb'))
''',
    "04_load_features.py": '''# Step 4: Load Features
def load_features():
    return pickle.load(open(os.path.join(WORKINGDIR, 'features.pkl'), 'rb'))
''',
    "05_load_captions.py": '''# Step 5: Load Captions & Mapping
def load_captions():
    with open(os.path.join(BASEDIR, 'captions.txt'), 'r') as f:
        captions_doc = f.read()
    mapping = dict()
    for line in tqdm(captions_doc.split('\n')):
        tokens = line.split(',')
        if len(tokens) < 2:
            continue
        image_id, caption = tokens[0].split('.')[0], tokens[1]
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping
''',
    "06_clean_mapping.py": '''# Step 6: Clean Captions
def clean_mapping(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].lower()
            caption = ''.join([c for c in caption if c.isalpha() or c.isspace()])
            caption = ' '.join([word for word in caption.split() if len(word) > 1])
            captions[i] = 'startseq ' + caption + ' endseq'
''',
    "07_tokenizer.py": '''# Step 7: Prepare Tokenizer
def create_tokenizer(mapping):
    all_captions = []
    for key in mapping:
        all_captions.extend(mapping[key])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer
''',
    "08_model_architecture.py": '''# Step 8: Define Model Architecture
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    plot_model(model, show_shapes=True)
    return model
''',
    "09_data_generator.py": '''# Step 9: Data Generator
def data_generator(keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for key in keys:
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
                    n += 1
                    if n == batch_size:
                        yield [np.array(X1), np.array(X2)], np.array(y)
                        X1, X2, y = [], [], []
                        n = 0
''',
    "10_training.py": '''# Step 10: Model Training Loop (Example)
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
''',
    "11_predict_caption.py": '''# Step 11: Caption Prediction
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text
''',
    "12_evaluation.py": '''# Step 12: Evaluation
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
''',
    "13_generate_and_show.py": '''# Step 13: Generate Caption & Show Image
def generate_caption_and_display(imagename, mapping, features, model, tokenizer, max_length):
    image_id = imagename.split('.')[0]
    img_path = os.path.join(BASEDIR, 'Images', imagename)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('----Actual Captions-----')
    for caption in captions:
        print(caption)
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('----Predicted Caption-----')
    print(y_pred)
    plt.imshow(image)
    plt.show()
''',
}


