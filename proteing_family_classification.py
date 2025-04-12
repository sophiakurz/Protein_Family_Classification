import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from umap import UMAP
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalMaxPooling1D, Conv1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Script started!")

MAX_SEQUENCE_LENGTH = 1000
BATCH_SIZE = 256
EPOCHS = 15
CHAR_LEVEL = True

def load_data():
    train = pd.read_csv('datafiles/train_data.csv')
    val = pd.read_csv('datafiles/val_data.csv')
    test = pd.read_csv('datafiles/test_data.csv')
    return train, val, test

def preprocess_data(df, tokenizer=None, le=None, is_test=False, max_len=MAX_SEQUENCE_LENGTH, char_level=CHAR_LEVEL):
    if not is_test and le is not None:
        y = le.transform(df['family_id'].values)
    else:
        y = None
    if tokenizer is None:
        tokenizer = Tokenizer(char_level=char_level)
        tokenizer.fit_on_texts(df['sequence'])
    X = tokenizer.texts_to_sequences(df['sequence'])
    X = pad_sequences(X, maxlen=max_len)
    if is_test:
        return X, tokenizer
    return X, y, tokenizer

def build_model(vocab_size, num_classes, input_length=MAX_SEQUENCE_LENGTH):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length),
        Conv1D(256, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def visualize_embeddings(model, X, y_true, output_dir=".", title_suffix=""):
    embedding_model = Model(inputs=model.input, outputs=model.layers[2].output)
    embeddings = embedding_model.predict(X)
    logging.info("Embeddings extracted: %s", embeddings.shape)
    
    # UMAP Visualization
    umap_proj = UMAP(n_components=2, random_state=42)
    umap_embeddings = umap_proj.fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    sc1 = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=y_true, cmap='viridis', s=10)
    plt.title(f"UMAP Projection of Protein Embeddings {title_suffix}")
    plt.colorbar(sc1)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    umap_file = os.path.join(output_dir, "umap_projection.png")
    plt.savefig(umap_file)
    plt.close()

    # t-SNE Visualization
    tsne_proj = TSNE(n_components=2, random_state=42, init='random')
    tsne_embeddings = tsne_proj.fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    sc2 = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=y_true, cmap='viridis', s=10)
    plt.title(f"t-SNE Projection of Protein Embeddings {title_suffix}")
    plt.colorbar(sc2)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    tsne_file = os.path.join(output_dir, "tsne_projection.png")
    plt.savefig(tsne_file)
    plt.close()

def plot_metrics(history, output_dir):
    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "train_val_accuracy.png"))
    plt.close()

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "train_val_loss.png"))
    plt.close()

def main():
    train, val, test = load_data()
    le = LabelEncoder()
    le.fit(train['family_id'])
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    X_train, y_train, tokenizer = preprocess_data(train, le=le)
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    X_val, y_val, _ = preprocess_data(val, tokenizer=tokenizer, le=le)
    X_test, _ = preprocess_data(test, tokenizer=tokenizer, is_test=True)
    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    vocab_size = len(tokenizer.word_index) + 1
    model = build_model(vocab_size, num_classes)
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    history = model.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val_cat), callbacks=callbacks, verbose=1)

    # Save accuracy and loss plots
    output_dir = os.path.dirname(os.path.abspath("submission.csv"))
    plot_metrics(history, output_dir)

    val_pred = model.predict(X_val)
    val_pred_classes = np.argmax(val_pred, axis=1)
    val_true_classes = np.argmax(y_val_cat, axis=1)
    val_accuracy = accuracy_score(val_true_classes, val_pred_classes)
    logging.info(f"Validation Accuracy: {val_accuracy:.4f}")
    
    visualize_embeddings(model, X_val, val_true_classes, output_dir=output_dir, title_suffix="(Validation Set)")

    test_pred = model.predict(X_test)
    test_pred_classes = np.argmax(test_pred, axis=1)
    predicted_family_ids = le.inverse_transform(test_pred_classes)
    submission = pd.DataFrame({
        'sequence_name': test['sequence_name'],
        'family_id': predicted_family_ids
    })
    submission.to_csv('submission.csv', index=False)
    logging.info("Submission file created.")

if __name__ == '__main__':
    main()