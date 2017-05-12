from keras.callbacks import EarlyStopping
from document_level_classification.constants import MODEL_OPTIMIZER, MODEL_LOSS, MODEL_METRICS, NB_EPOCHS, BATCH_SIZE, \
    PREDICTION_TYPE, LOGS_DIR, MODEL_DIR

from helpers.model_utils import save_trained_model



from helpers.helper_functions import get_time_format, log_session
import time

def train(model, model_info, data, save_model=False, extra_info=None):
    """
    
    :param model: Model object, the model that is going to be trained
    :param model_info: list of strings, information about parameters etc.
    :param data: dictionary with validation, test and train set, ie {'x_train':[], 'y_trian:[], 'x_val':[], 'y_val':[]}
    :param extra_info: 
    :return: 
    """


    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    model.compile(optimizer=MODEL_OPTIMIZER,
                  loss=MODEL_LOSS,
                  metrics=MODEL_METRICS)

    # Time
    start_time = time.time()

    print('\nCommence training %s model' % model.name)
    history = model.fit(data['x_train'], data['y_train'],
                        validation_data=[data['x_val'], data['y_val']],
                        epochs=NB_EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=[early_stopping]).history

    seconds = (time.time() - start_time)
    training_time = get_time_format(seconds)
    print "Training time: %s" % training_time

    # Evaluate on test set
    test_results = model.evaluate(data['x_test'], data['y_test'], batch_size=BATCH_SIZE)

    """PARAMS => log_session(log_dir, model, history, training_time, num_train, num_val, optimizer, batch_size, max_epochs,
                                        test_results, model_info=None, extra_info=None, max_sequence_length=None):"""
    log_session(log_dir=LOGS_DIR,
                model=model,
                history=history,
                training_time=training_time,
                num_train=len(data['x_train']),
                num_val=len(data['x_val']),
                num_test=len(data['x_test']),
                optimizer=MODEL_OPTIMIZER,
                batch_size=BATCH_SIZE,
                max_epochs=NB_EPOCHS,
                test_results=test_results,
                model_info=model_info,
                extra_info=extra_info,
                max_sequence_length=data['x_train'].shape[1])

    if save_model:
        save_trained_model(model, MODEL_DIR, MODEL_OPTIMIZER)







