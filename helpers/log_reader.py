import os
import pandas as pd

def get_logs(path, modeltype, data_description='training_statistics', model_file=None):

    data_description = " ".join(data_description.split('_'))
    if not model_file:
        model_file = list(filter(lambda x: ".txt" in x, os.listdir(os.path.join(path, modeltype))))[-1]
    path = path + modeltype + "/" + model_file
    f = open(path, 'r')

    head_type = []
    all_values = []
    head_entry = False
    value_entry = False
    for line in f:
        if data_description in line.lower():
            head_entry = True
            continue
        if head_entry:
            head_type.extend(line.split())
            head_entry = False
            value_entry = True
            continue
        if value_entry:
            values = line.split()
            if len(values) == 0:
                break
            else:
                all_values.append(values)

    # printing test
    # for x in all_values:
    #     print x
    #all_values = zip(*all_values)[1:]
    #
    # for x in all_values:
    #     print x

    all_values = zip(*all_values)[1:]

    data_dic = dict(zip(head_type, all_values))
    return data_dic



if __name__ == "__main__":
    __dir__ = "../logs/document_level_classification/"
    dirs = filter(lambda x: "base" in x, os.listdir(__dir__))
    stats = dict()
    for model_type in dirs:
        log = get_logs(__dir__, model_type)
        data_frame = pd.DataFrame(log)
        min_value = data_frame['val_loss'].min()
        stats[model_type] = min_value
        print '{:<20}{:<1}'.format(model_type, round(float(min_value), 3))
        #print model_type, "      ", min_value
        # formatting string



    #print min(stats, key=stats.get)
    #print stats




