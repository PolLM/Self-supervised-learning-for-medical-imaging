#%%
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pylab as plt
import numpy as np
import pickle 
import os
import glob

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

def load_log_data(path, logname = False, metric = "Acc/eval", num_epochs = "num_epochs_sup"):
    base_path = os.path.dirname(path)
    config = load_pickle_dict(base_path)
    df = tflog2pandas(base_path)
    df = df[df['metric'].str.contains(metric)]
    batches_per_epoch = int(len(df)/config[num_epochs])  
    df = df.groupby(np.arange(len(df)) // batches_per_epoch).mean()
    df["step"] = df.index
    if logname:
        df = df.rename(columns={"step": "step_"+logname,
                                "value": "value_"+logname})
    return(df)  

def load_log_data_unsup(path, logname = False, metric = "Acc/eval", num_epochs = "num_epochs"):
    base_path = os.path.dirname(path)
    #config = load_pickle_dict(base_path)
    df = tflog2pandas(base_path)
    df = df[df['metric'].str.contains(metric)]
    #batches_per_epoch = int(len(df)/config[num_epochs])  
    #df = df.groupby(np.arange(len(df)) // batches_per_epoch).mean()
    df["step"] = df.index
    if logname:
        df = df.rename(columns={"step": "step_"+logname,
                                "value": "value_"+logname})
    return(df) 

def many_logs2pandas(path, contains = False, not_contains = False,  mode = 'sup', metric = "Acc/eval"):

    if os.path.isdir(path):
        event_paths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                
                if "events.out.tfevents" in file:
                    if contains:
                        if contains not in root:
                            continue
                        
                    if not_contains:
                        if not_contains in root:
                            continue
                    event_paths.append(os.path.join(root, file))
                    
    elif os.path.isfile(path):
        event_paths = [path]

    all_logs = pd.DataFrame()
    for path in event_paths:
        try:
            if mode == 'sup':
                log = load_log_data(path, logname= path.split("\\")[-2], metric = metric)
            else:
                log = load_log_data_unsup(path, logname= path.split("\\")[-2], metric = metric)
        except:
            print(f"Excepting: {path}")
            log = None
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs

def get_last_epoch_metric(all_logs):
    last_epoch_metric = {}
    for col in all_logs.columns:
        if "step" in col:
            x_name = col
            y_name = "value_"+col[5:]
            last_epoch_metric[y_name] = list(all_logs[y_name][all_logs[x_name] == np.max(all_logs[x_name])])
    #last_df = pd.DataFrame(last_epoch_metric)
    return(last_epoch_metric)

def load_pickle_dict(path, filername = "config.p"):
    with open(path+"\\"+filername, 'rb') as handle:
        config = pickle.load(handle)
    return(config)

if __name__ == "__main__":
    path = r"D:\Documents\GitHub\aidl2022_final_project\runs\final_trainings\scan_supervised_2"
    all_logs = many_logs2pandas(path, contains=False)

    last_epoch_metric = get_last_epoch_metric(all_logs)
    for col in all_logs.columns:
        if "step" in col:
            x_name = col
            y_name = "value_"+col[5:]
            plt.scatter(all_logs[x_name], all_logs[y_name], label= col[5:], marker='>')
            plt.plot(all_logs[x_name], all_logs[y_name], ls = '--')
    #plt.ylim(0.9,1)  
    #plt.legend()      
    plt.grid()
    plt.show()

# %%
