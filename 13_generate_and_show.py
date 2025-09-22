# Step 13: Generate Caption & Show Image
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
