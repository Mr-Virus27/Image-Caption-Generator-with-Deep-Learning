# Step 4: Load Features
def load_features():
    return pickle.load(open(os.path.join(WORKINGDIR, 'features.pkl'), 'rb'))
