# -*- coding: utf-8 -*-
import pandas as pd
import sys
import os
import pickle

def process_data(install, home, active):
    columns = install.columns.values.tolist()

    user_app_pair_install = pd.melt(
        install.reset_index(), 
        id_vars=['index'], 
        value_vars=columns)

    user_app_pair_active = pd.melt(
        active.reset_index(), 
        id_vars=['index'], 
        value_vars=columns)

    user_app_pair_active = user_app_pair_active.\
        set_index(['index','variable']).\
            rename(columns={'value':'active'})

    # Select rows that have the app installation
    user_app_pair_install = user_app_pair_install.loc[user_app_pair_install['value'] != 0]

    user_app_pair_install_home = user_app_pair_install.\
        set_index('index').\
            join(home,how='left').\
                join(install, how='left')

    user_app_pair_install_home = user_app_pair_install_home.\
        reset_index().\
            set_index(['index','variable'])

    df = user_app_pair_install_home.join(user_app_pair_active,how = 'left')

    return user_app_pair_install, user_app_pair_install_home, df


def main(args):
    _DATA_PICKLE_DIR = 'data/0210-10kSample-Data/pickle_file/'

    install = pd.read_pickle(
        os.path.join(
            _DATA_PICKLE_DIR,
            'metrics_user_app_install'))
    install[install>1] = 1

    home = pd.read_pickle(
        os.path.join(
            _DATA_PICKLE_DIR,
            'metrics_user_site_home'))

    active = pd.read_pickle(
        os.path.join(
            _DATA_PICKLE_DIR,
            'metrics_user_app_active'))
    active[active>1] = 1

    target_app = pd.read_pickle(
        os.path.join(
            _DATA_PICKLE_DIR,
            'target_app_list')).index.values.tolist()

    install = install[target_app]
    active = active[target_app]

    _, _, df = process_data(install, home, active)

    #print(df.describe())
    df.to_pickle(os.path.join(_DATA_PICKLE_DIR,'user_app_pair_df'))

    #print(df.shape)

if __name__ == "__main__":
    main()
