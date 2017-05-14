from keras.callbacks import EarlyStopping
from document_level_classification.constants import MODEL_OPTIMIZER, MODEL_LOSS, MODEL_METRICS, NB_EPOCHS, BATCH_SIZE, \
    PREDICTION_TYPE, LOGS_DIR, MODEL_DIR

from helpers.model_utils import get_model_checkpoint, save_trained_model, get_precision_recall_f_score



from helpers.helper_functions import get_time_format, log_session
import time

def train(model, model_info, data, save_model=False, extra_info=None):
    """
    
    :param model: Model object, the model that is going to be trained
    :param model_info: list of strings, information about parameters etc.
    :param data: dictionary with validation, test and txt set, ie {'x_train':[], 'y_trian:[], 'x_val':[], 'y_val':[]}
    :param extra_info: 
    :return: 
    """


    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    model.compile(optimizer=MODEL_OPTIMIZER,
                  loss=MODEL_LOSS,
                  metrics=MODEL_METRICS)

    callbacks = [early_stopping]

    if save_model:
        callbacks.append(get_model_checkpoint(model.name, MODEL_DIR, MODEL_OPTIMIZER))

    # Time
    start_time = time.time()

    print('\nCommence training %s model' % model.name)
    history = model.fit(data['x_train'], data['y_train'],
                        validation_data=[data['x_val'], data['y_val']],
                        epochs=NB_EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=callbacks).history

    seconds = (time.time() - start_time)
    training_time = get_time_format(seconds)
    print "Training time: %s" % training_time

    # Compute prf for val set
    prf_val = get_precision_recall_f_score(model, data['x_val'], data['y_val'], PREDICTION_TYPE)

    # Evaluate on test set, if supplied
    if 'x_test' in data:
        test_results = model.evaluate(data['x_test'], data['y_test'], batch_size=BATCH_SIZE)
        prf_test = get_precision_recall_f_score(model, data['x_test'], data['y_test'], PREDICTION_TYPE)
        num_test = len(data['x_test'])
    else:
        test_results = None
        num_test = 0
        prf_test = None

    """PARAMS => log_session(log_dir, model, history, training_time, num_train, num_val, optimizer, batch_size, max_epochs,
                                        test_results, model_info=None, extra_info=None, max_sequence_length=None):"""
    log_session(log_dir=LOGS_DIR,
                model=model,
                history=history,
                training_time=training_time,
                num_train=len(data['x_train']),
                num_val=len(data['x_val']),
                num_test=num_test,
                optimizer=MODEL_OPTIMIZER,
                batch_size=BATCH_SIZE,
                prf_val=prf_val,
                max_epochs=NB_EPOCHS,
                test_results=test_results,
                model_info=model_info,
                extra_info=extra_info,
                max_sequence_length=data['x_train'].shape[1],
                prf_test=prf_test)

    if save_model:
        save_trained_model(model, MODEL_DIR, MODEL_OPTIMIZER)







