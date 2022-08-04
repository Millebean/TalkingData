import pandas as pd
import numpy as np
import pickle
import os
import argparse

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, accuracy_score, make_scorer

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names


def merge_data(_DATA_PICKLE_DIR):
    install = pd.read_pickle(
        os.path.join(
            _DATA_PICKLE_DIR,
            'metrics_user_app_install'))
    
    home = pd.read_pickle(
        os.path.join(
            _DATA_PICKLE_DIR,
            'metrics_user_site_home'))

    df = install.join(home, how= 'left').sort_index()

    df.to_pickle(os.path.join(_DATA_PICKLE_DIR, 'install_home_active_packed'))

    return df

def read_data(_DATA_PICKLE_DIR):
    install = pd.read_pickle(
        os.path.join(
            _DATA_PICKLE_DIR,
            'metrics_user_app_install')).sort_index()

    df = pd.read_pickle(
        os.path.join(
            _DATA_PICKLE_DIR,
            'install_home_active_packed')).sort_index()    

    active = pd.read_pickle(
        os.path.join(
            _DATA_PICKLE_DIR,
            'metrics_user_app_active')).sort_index()
    active[active>1] = 1

    return df, active, install

def app_with_highst_installation_rate(install_thr, active_thr, install, active):
    i_list = install.mean().to_frame('install_rate').loc[install.mean()>install_thr]
    a_list = active.mean().to_frame('active_rate').loc[active.mean()>active_thr]
    return i_list.join(a_list, how = 'inner')

def train_xgboost(X_train, y_train, X_test, y_test):
    scale = 1/y_train.mean()
    print('The parameter scale_pos_weight is set to be %.2f'% (scale))

    model = xgb.XGBClassifier(n_jobs=-1,scale_pos_weight=scale)
    model.fit(X_train, y_train)
    #pickle.dump(model, open(model_dir + "benchmark_xgboost_" + app + ".pickle.dat", "wb"))

    y_score = model.predict_proba(X_test)[:,1]
    return y_score

def train_RF(X_train,y_train,X_test,y_test):
    parameters = {'n_estimators':[3, 5, 10, 15, 20, 50, 100], 
                    'min_samples_split':[2, 5, 10, 15, 20],
                    'n_jobs': [-1],
                    'class_weight': ['balanced']
                    }
    auc_score_funtion = make_scorer(roc_auc_score)
    RF = RandomForestClassifier() 
    clf = GridSearchCV(RF, parameters, scoring=auc_score_funtion)
    clf.fit(X_train, y_train) 
    print(clf.best_params_)
    y_score = clf.predict_proba(X_test)[:,1]
    return y_score

def train_DeepFM(X, y, sparse_cols): 
    # Rename the columns
    cols = X.columns.tolist()
    cols_map = {name:str(i) for i,name in enumerate(cols)}
    X = X.rename(columns=cols_map)

    # Record sparse feature names and dense feature names
    sparse_features = [cols_map[key] for key in sparse_cols]
    dense_features = [value for value in cols_map.values() if value not in sparse_features]

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=X[feat].nunique(),embedding_dim=4)
                    for feat in sparse_features] + [DenseFeat(feat, 1,)
                          for feat in dense_features]
    #print(fixlen_feature_columns)
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns    
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    seed = 7; test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    train_model_input = {name:X_train[name] for name in feature_names}
    test_model_input = {name:X_test[name] for name in feature_names}

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, y_train.values, batch_size=256, epochs=10, verbose=2, validation_split=0.2)
    y_score = model.predict(test_model_input, batch_size=256)  
    return y_score, y_test

def training(app_list, model_dir, X_df, y_df, install, model_type, mode):
    if model_dir:   
        try:
            os.mkdir(model_dir)
        except:
            pass
    
    performance_for_all_apps = app_list.reindex(
        columns = app_list.columns.tolist()+['precision', 'recall', 'f1', 'thres', 'accuracy', 'AUC'])  

    for i, app in enumerate(app_list.index.tolist()):
        if mode == 'on_off':
            df = X_df.join(y_df[app].to_frame().rename(columns={app:'active'}), how = 'left')
            X = df.drop('active', 1); y = df['active']
        elif mode == 'on_only':
            X = install; y = y_df[app]

        seed = 7; test_size = 0.2

        if model_type == 'xgboost':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed)
            y_score = train_xgboost(X_train,y_train,X_test,y_test)
        elif model_type == 'DeepFM':
            sparse_cols = install.columns.tolist()
            #y_score = train_DeepFM(X_train[app_list.index.tolist()],y_train,\
                # X_test[app_list.index.tolist()],y_test)
            y_score, y_test = train_DeepFM(X, y, sparse_cols)
            #pickle.dump((ypred,y_test), open(model_dir + "prediction.pkl", "wb"))
        elif model_type == 'RF':    

            y_score = train_RF(X_train,y_train,X_test,y_test)

        try:
            precision, recall, thres = precision_recall_curve(y_test, y_score)
            f1_scores = 2*recall*precision/(recall+precision)
            performance_for_all_apps.loc[app, 'f1'] = np.nanmax(f1_scores)
            performance_for_all_apps.loc[app, 'precision'] = precision[np.nanargmax(f1_scores)]
            performance_for_all_apps.loc[app, 'recall'] = recall[np.nanargmax(f1_scores)]
            performance_for_all_apps.loc[app, 'thres'] = thres[np.nanargmax(f1_scores)]
            accuracy = accuracy_score(y_test, np.where(y_score > thres[np.nanargmax(f1_scores)], 1, 0))
            performance_for_all_apps.loc[app, 'accuracy'] = accuracy
        except:
            print("Precision and recall is not defined in this app")   
            pass      

        try:  
            auc = roc_auc_score(y_test, y_score)
            performance_for_all_apps.loc[app, 'AUC'] = auc
        except ValueError:
            auc = None
        
        print(performance_for_all_apps.loc[app])
        
        #plot_cache.append((app, best_precision, best_recall, best_thres, accuracy, auc))
        if auc and accuracy:       
            #print("Roc_auc: %.2f%%" % (auc*100) )
            print("Accuracy: %.2f%%, Roc_auc: %.2f%%" % \
                (accuracy*100.0, auc*100.0))
        elif auc:
            print("Accuracy: NA, Roc_auc: %.2f%%" % \
                (auc*100.0))   
        elif accuracy:
            print("Accuracy: %.2f%%, Roc_auc: NA" % \
                (accuracy*100.0))        

        if (i+1) % 10 == 0:
            print('='*10 + 'Finish training %i apps'%(i+1) +'='*10 )
    
    pickle.dump(performance_for_all_apps, open(model_dir + "result.pkl", "wb"))
    performance_for_all_apps.to_csv(model_dir + 'prediction.csv')
    return 

def main():
    """
    Sample input: 
    python benchmark.py --modeltype 'RF'
    python benchmark.py  --modeldir '../model/DeepFM/' --modeltype 'DeepFM'
    python benchmark.py  --modeldir '../model/DeepFM_on_off/' --modeltype 'DeepFM'
    """
    parser = argparse.ArgumentParser(description='Arguements')
    parser.add_argument('--install_thr', type = float, default = 0.01,
                    help='Popularity threshold: app installation rate')
    parser.add_argument('--active_thr', type = float, default = 0.0008,
                    help='Popularity threshold: app active user rate')
    parser.add_argument('--datadir', type = str, default = '../data/0210-10kSample-Data/pickle_file/',
                    help='Directory of data folder')
    parser.add_argument('--m', type = bool, default = False,
                    help='Merge data before reading it')
    parser.add_argument('--modeldir', type = str, default = '../model/xgboost/',
                    help='Directory of model folder')
    parser.add_argument('--modeltype', type = str, default = 'xgboost',
                    help='Type of model to use')   
    parser.add_argument('--mode', type = str, default = 'on_off',
                    help='Data to use: online only or online offline combined')                   

    args = parser.parse_args()  
    print(args)

    if args.datadir:
        merge_data(args.datadir) 
    
    X_df, y_df, install = read_data(args.datadir) 

    app_list = app_with_highst_installation_rate(args.install_thr, args.active_thr, install, y_df)
    print('Select apps with installation rate over %.2f%%; The number of apps is %i'\
        %(args.install_thr*100, app_list.shape[0]))

    training(app_list, args.modeldir, X_df, y_df, install, args.modeltype, args.mode)

    return 

if __name__ == "__main__":           
    main()
