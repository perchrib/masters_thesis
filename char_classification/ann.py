from char_classification.text_preprocessing import prepare_dataset
from char_classification.models import get_char_model, get_char_model_2
from keras.callbacks import EarlyStopping
from char_classification.constants import MODEL_OPTIMIZER, MODEL_LOSS, MODEL_METRICS, NB_EPOCHS, BATCH_SIZE


def train():
    # Load dataset
    x_train, y_train, meta_train, x_val, y_val, meta_val, char_index, labels_index = prepare_dataset()

    model = get_char_model_2(len(labels_index))

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    model.compile(optimizer=MODEL_OPTIMIZER,
                  loss=MODEL_LOSS,
                  metrics=MODEL_METRICS)

    print('\nCommence training model')
    model.fit(x_train, y_train,
              validation_data=[x_val, y_val],
              nb_epoch=NB_EPOCHS,
              batch_size=BATCH_SIZE,
              shuffle=True,
              callbacks=[early_stopping])


train()