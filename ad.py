"""
Auxiliary functions to "Functional activity level reported by an informant is an early predictor of Alzheimer's disease" paper

(C) MMIV-ML MCI group.

Created: 2022.10.12
Updated: 2022.10.12
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path
from sklearn import metrics
from sklearn.base import clone



def dropcol_importances(rf, X_train, y_train, X_test, y_test, random_state=42, groups=[], verbose=True, precission=2):
    """
    C: 2021.06.23 / U: 2021.06.23
    """
    column_list = X_train.columns
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()
    y_test = y_test.copy()
    
    ### INFO PART ##############################################
    all_features, all_feature_names = _get_feature_group_info(column_list, groups, verbose)    
    ### END OF INFO PART ##############################################    
   
    
    rf_ = clone(rf)
    rf_.random_state = random_state
    rf_.fit(X_train, y_train)
    y_test_pred = rf_.predict(X_test)
    # baseline scores
    f1_baseline, acc_baseline, recall_baseline, prec_baseline = _get_4_scores(y_test, y_test_pred)
    # list for score drops
    f1_list, acc_list, recall_list, prec_list = [], [], [] ,[]  
    
    
    for k, cols in enumerate(all_features):
        X_train_drop = X_train.drop(cols, axis=1)
        X_test_drop = X_test.drop(cols, axis=1)
        
        rf_ = clone(rf)
        rf_.random_state = random_state
        rf_.fit(X_train_drop, y_train)
        
        y_test_drop_pred = rf_.predict(X_test_drop)
        
        # selectd feature permutation scores
        f1_f, acc_f, recall_f, prec_f = _get_4_scores(y_test, y_test_drop_pred)  
        
        # difference between baseline and feature scores        
        f1_list.append(f1_baseline - f1_f)
        acc_list.append(acc_baseline - acc_f)
        recall_list.append(recall_baseline - recall_f)
        prec_list.append(prec_baseline - prec_f)        
        
    
    df = pd.DataFrame.from_dict({'f1':f1_list, 'acc':acc_list, 'recall':recall_list, 'prec':prec_list})
    df.index = all_feature_names.keys()    
    
    return df.round(precission), all_feature_names

def rename_columns(df, dc_names, verbose=True):
    """
    Rename column names in a df. Function returns a COPY of original df with a new column names.
    
    Parameters:
    --------------------------
    df - a df to change column names,
    dc_names - dictionary with old names (as keys) and new names (as values),
    verbose - prints out old and new column names.
    
    
    Usage:
    --------------------------
    dct_names = {'AGE': 'age', 'FAQ': 'faq'}
    bl = rename_columns(bl, dc_names=dct_names)
        
    C: 2021.07.30 / U: 2021.07.30
    """
    
    keys_new = dc_names.keys()
    keys_old = df.columns
    
    for k in keys_new:        
        if k not in keys_old:
            print(f'Wrong column name: {k}')
    print()
            
            
    if verbose: print(f'OLD names:\n{df.columns}\n')
    df_new = df.rename(columns=dc_names)    
    if verbose: print(f'NEW names:\n{df_new.columns}\n')
    
    return df_new

def _print_group_names(features_dct):
    """
    C: 2021.06.23 / U:2021.06.23
    """
    [print(f'{k}:{v}') for k,v in features_dct.items() if isinstance(v, list)]

def plot_permuted_features_single(df, performance_name, file_name_prefix, type, save=True, results_dir=Path().cwd(), figsize=(22,12), title_suffix='' ):
    """
    type = string, one of these values: empty/-drop/-random
    performance_name - name of score to plot, one out of four, eg. "acc"
    title_suffix - some additional text to display in the main title
    
    C:2021.06.23 / U:2022.03.30
    """

    fig, ax = plt.subplots(1,1, figsize=figsize)
    #axs = ax.flat[:]

    #for ax, f in zip(axs, df.columns.to_list()):
    sns.barplot(x=df[performance_name], y=df.index, ax=ax)

    # Add labels to your graph
    ax.set_xlabel(ax.get_xlabel(), fontsize=18, fontweight='bold')
    ax.xaxis.set_label_position('top')

    #ax.set_ylabel('Feature(s)', fontsize=18)
    #ax.set_title(f, fontsize=20, pad=20, fontweight='bold')
    ax.tick_params(labelsize=22)
    ax.grid(True)

    plt.subplots_adjust(hspace=0.25)
    #plt.suptitle(f'Feature importance. {title_suffix}', fontsize=26, fontweight='bold')
    
    if save:
        file_name_prefix_ext = f'{file_name_prefix}-{type}-features.pdf'
        file_name_prefix_path = results_dir / file_name_prefix_ext
        plt.savefig(file_name_prefix_path)
        print(f'Shuffle [group] feature(s) saved to:\n\t\t{file_name_prefix_path}\n')

    plt.show()


def rename_columns(df, dc_names, verbose=True):
    """
    Rename column names in a df. Function returns a COPY of original df with a new column names.
    
    Parameters:
    --------------------------
    df - a df to change column names,
    dc_names - dictionary with old names (as keys) and new names (as values),
    verbose - prints out old and new column names.
    
    
    Usage:
    --------------------------
    dct_names = {'AGE': 'age', 'FAQ': 'faq'}
    bl = rename_columns(bl, dc_names=dct_names)
        
    C: 2021.07.30 / U: 2021.07.30
    """
    
    keys_new = dc_names.keys()
    keys_old = df.columns
    
    for k in keys_new:        
        if k not in keys_old:
            print(f'Wrong column name: {k}')
    print()
            
            
    if verbose: print(f'OLD names:\n{df.columns}\n')
    df_new = df.rename(columns=dc_names)    
    if verbose: print(f'NEW names:\n{df_new.columns}\n')
    
    return df_new

def _get_4_scores(y_true, y_pred):
    """
    Get all scores.
    
    C: 2021.05.20 / U: 2021.05.20
    """    
    f1 = metrics.f1_score(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred)
    return f1, acc, recall, prec

def _get_feature_group_info(column_list, groups, verbose):
    """
    C: 2021.06.23 / U:2021.06.23
    """
    feat_list = []
    for g in groups:
        feat_list.extend(g)
        
    # check feature names (spelling mistakes)
    no  = [n for n in feat_list if n not in column_list]
    if no:
        print(f'\nWrong names ({len(no)}/{len(feat_list)}):\n\t{no} ')
        print('\n*** Fix it NOW!!!\n\t\tha ha ha ;D\n\n')
        print(f'Valid feature names:\n{column_list.to_list()}')
        return

    # divide by groupS and 'single' features
    singles = set(column_list)    
    for g in groups:
        singles = singles.difference(set(g))        
    # list wiht all features (groups and singles)
    all_features = groups + list(sorted(singles))   
    
    # an extra dict to replace list of features (a group) with an name G_0, G_1, ...
    all_feature_names = {}
    for k, f in enumerate(all_features):
        if isinstance(f,list):
            all_feature_names[f'Group_{k}']=f
        else:
            all_feature_names[f]=f       
    # info (optonal)
    if  verbose:    
        # print info
        print(f'All features:\n\t{column_list.to_list()}')    
        print('\nFeature groups:')
        for g in groups:
            print(f'\t{g}')
        print(f'\nSingle features:\n\t{sorted(list(singles))}')
        print('\n\n')
    
    return all_features, all_feature_names

def shuffle_features_with_groups(rf, X, y, groups=[], precission=2, verbose=True, random_state=None,
                                 repetitions=100, sortBy=None, ascending=True):
    """
    Feature permutation. Function permutes featureas from a X set, some of them can be joined and permuted in groups.
    
    Parameters:
    --------------------
    groups - a nested list. Contains grouped feature namse in separate lists e.g.
                                                    gropus= [['a1','a2','a3'], ['b1','b2','b3','b4'], ['c1','c2']]
    random_state - int number or None. If None, randomly selected random seed value (each run gives different result).
    repetions - nr of repetitions to average the restult. If repetitions > 0 then random state is automatically set to None. This will print random feature permutation (random values).
    sortBy - a feature name to sort values by : 'f1'/'acc'/'recall'/'prec'
    ascending - wheter sortBy in ascending or descending order (True/False).
    
    
    C: 2021.05.01 / U: 2021.06.24
    """
    column_list = X.columns
    X = X.copy()
    y = y.copy()
    
    if repetitions > 0:
        rep = repetitions # shorter name
        random_state = None # random seed (each run different)
        print(f'Repetition(s) = {rep+1}\nAveraging mode!\nrandom_state = {random_state}\n')
    else:
        rep = 1
        print(f'Repetition(s) = {rep+1}\nSingle permutation mode\nrandom_state = {random_state}\n')
    
    ### INFO PART ##############################################
    all_features, all_feature_names = _get_feature_group_info(column_list, groups, verbose)
    #print(all_features)
    #print(all_feature_names)
    ### END OF INFO PART ##############################################    
    
    # prediction
    y_pred = rf.predict(X)    
    # baseline scores
    f1_baseline, acc_baseline, recall_baseline, prec_baseline = _get_4_scores(y, y_pred)
    # list for score drops


    f1_list, acc_list, recall_list, prec_list = [],[],[],[]        
    for k, cols in enumerate(all_features):        
        # permutation of the selected column(s)
        save = X[cols].copy()
        #X[cols] = np.random.permutation(X[cols])
        
        f1_a, acc_a, recall_a, prec_a = np.zeros(rep), np.zeros(rep), np.zeros(rep), np.zeros(rep)
        # repetition loop
        for r in range(rep):
            X[cols] = np.random.RandomState(random_state).permutation(X[cols])

            y_pred_f = rf.predict(X)               
            # arrays for selectd feature permutation scores
            f1_f, acc_f, recall_f, prec_f = _get_4_scores(y, y_pred_f)        
            # difference between baseline and feature scores        
            f1_a[r] = f1_baseline - f1_f
            acc_a[r] = acc_baseline - acc_f
            recall_a[r] = recall_baseline - recall_f
            prec_a[r] = prec_baseline - prec_f
        
        f1_list.append(f1_a.mean())
        acc_list.append(acc_a.mean())
        recall_list.append(recall_a.mean())
        prec_list.append(prec_a.mean())
        # restore the initial column value
        X[cols] = save 
        
        
    df = pd.DataFrame.from_dict({'f1':f1_list, 'acc':acc_list, 'recall':recall_list, 'prec':prec_list})
    df.index = all_feature_names.keys()       
    
    if sortBy:
        df = df.sort_values(sortBy, ascending=ascending)        
    return df.round(precission), all_feature_names


def plot_single_feature_importnce(df, file_name_prefix, figsize=(20,10), orientation='h', save=True, results_dir=Path().cwd()):
    """

    C: 2021.05.04 / U: 2021.05.04
    """

    fig, ax = plt.subplots(figsize=figsize)

    if orientation == 'h':
        sns.barplot(x=df, y=df.index, ci='sd', capsize=.2,  orient=orientation)
    else:
        sns.barplot(x=df.index, y=df, ci='sd', capsize=.2,  orient=orientation)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45) 


    #sns.barplot(x=df, y=df.index, ci='sd')
    # Add labels to your graph
    ax.set_xlabel('Feature Importance Score', fontsize=24)
    ax.set_ylabel('Features',fontsize=24)
    ax.set_title("RF Feature Importance", fontsize=24, fontweight='bold', pad=20)
    ax.tick_params(labelsize=14)

    if save:
        file_name_prefix_ext = f'{file_name_prefix}-TEST-feat-importance-{orientation}.png'
        file_name_prefix_path = results_dir / file_name_prefix_ext
        plt.savefig(file_name_prefix_path)
        print(f'Mean featue importacne plot saved to:\n\t\t{file_name_prefix_path}\n')
    
    plt.grid()
    plt.show()


def rename_columns(df, dc_names, verbose=True):
    """
    Rename column names in a df. Function returns a COPY of original df with a new column names.
    
    Parameters:
    --------------------------
    df - a df to change column names,
    dc_names - dictionary with old names (as keys) and new names (as values),
    verbose - prints out old and new column names.
    
    
    Usage:
    --------------------------
    dct_names = {'AGE': 'age', 'FAQ': 'faq'}
    bl = rename_columns(bl, dc_names=dct_names)
        
    C: 2021.07.30 / U: 2021.07.30
    """
    
    keys_new = dc_names.keys()
    keys_old = df.columns
    
    for k in keys_new:        
        if k not in keys_old:
            print(f'Wrong column name: {k}')
    print()
            
            
    if verbose: print(f'OLD names:\n{df.columns}\n')
    df_new = df.rename(columns=dc_names)    
    if verbose: print(f'NEW names:\n{df_new.columns}\n')
    
    return df_new


def plot_confusion_matrix_TEST(conf_matrix_test, conf_matrix_test_prc, file_name_number, file_name_prefix, results_dir=Path().cwd(), save=True):
    """
    C: 2021.06.23 / U: 2021.06.23
    """
    title = f'Confusion matrix - TEST ({file_name_number})'
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect(aspect=1)
    lab = ['sMCI', 'cAD']
    res = sns.heatmap(conf_matrix_test, annot=True, xticklabels=lab,
                      yticklabels=lab, ax=ax,annot_kws={"fontsize":20}, fmt='.1f')
    ax.tick_params(axis='both', which='major', labelsize=18)
    _ = ax.set_title(title, size=24, fontweight='bold')


    for t,p in zip(res.texts, conf_matrix_test_prc.flat):
        p = np.asarray(np.round(p,0), int)
        t.set_text(t.get_text() + f' ({p}%)')

    file_name_prefix_ext = f'{file_name_prefix}-conf-matrix-TEST.png'
    file_name_prefix_path = results_dir / file_name_prefix_ext
    
    if save:
        plt.savefig(file_name_prefix_path)
    plt.show()


def confusion_matrix_coefficients_TPTNFPFN(X, y_true, y_pred):
    """
    Calculate usion matrix coefficients.
    
    Parameters:
    -------------
    X - test or validation (in e.g. CV10) set
    
    C: 2021.05.18./ U: 2021.05.18
    """

    X_extended = X.copy()
    X_extended['y_true_'] = np.where(y_true == 1, 'cAD', 'sMCI')
    X_extended['y_pred_'] = np.where(y_pred == 1, 'cAD', 'sMCI')
    
    # https://www.statology.org/compare-two-columns-in-pandas/
    conditions = [(X_extended['y_true_'] == 'cAD') &  (X_extended['y_pred_'] == 'cAD'),
                 (X_extended['y_true_'] == 'sMCI') &  (X_extended['y_pred_'] == 'sMCI'),
                 (X_extended['y_true_'] == 'sMCI') &  (X_extended['y_pred_'] == 'cAD'),
                 (X_extended['y_true_'] == 'cAD') &  (X_extended['y_pred_'] == 'sMCI')]
    choices = ['TP', 'TN', 'FP', 'FN']
    X_extended['CM_pred_'] = np.select(conditions, choices, default='Error')
    
    return X_extended


def link_prediction_results_with_other_subject_features(bl_table, predictions_df, cols2, filename='', save=True, results_dir=Path().cwd()):
    """
    Links prediction results with all other subject features.
    
    Parameters:
    ---------------
    bl_table - the bl table with all feature subjects to read from.
    predictions_df - a table with prediction results (i.e. X_extended with y_pred, y_true_, CM_pred_ )
    cols2 - an extra columns to select from `prediction_df` e.g. in CV process ([f'CV{FOLDS}F_',  f'CV{FOLDS}_Usage_'])
    
    C: 2021.05.18./ U: 2021.05.18
    """
    
    # get 'TRAIN' subset from the loaded df (train + test)
    bl_pred = bl_table.loc[predictions_df.index]
    # select some columns from misclassified subject df
    cols = ['y_true_', 'y_pred_', 'CM_pred_'] + cols2
    # merge both tables by index
    bl_pred = bl_pred.merge(predictions_df[cols], how='left', left_index=True, right_index=True, indicator=f'MERGE_predictions_')
    print(f'\nSubjects in the predictions table: {bl_pred.shape[0]}\n')
    
    if save:     
        bl_predictions_name = results_dir / filename
        bl_pred.sort_values(by=['RID'], inplace=True) 
        bl_pred.to_csv(bl_predictions_name, index=True)
        
        print(f'Predictions have been saved to a file:')
        print(f'\t\t{bl_predictions_name}')
        
    return bl_pred


def included_feature_info(df, pattern='adni-adas-neuro-gds-faq-long-cross-_'):
    """
    Create a df with names types (e.g. _adas, _neuro, _) present in df's columns. It serves to check the curret content of tables in terms of included features.
    
    pattern: feature names with dash among them.
    
    
    Included patterns:
    - adni,
    - adas,
    - neuro,
    - gds,
    - faq,
    - long,
    - cross,
    - ours.
    
    
    The last update: Listing of 'faq' features.    
    C: 2021.03.10 / U: 2021.10.04
    """
    
    adas_lst = sorted([c for c in df.columns if c.endswith('_adas')])
    neuro_lst = sorted([c for c in df.columns if c.endswith('_neuro')])
    gds_lst = sorted([c for c in df.columns if c.endswith('_gds')])         
    faq_lst = sorted([c for c in df.columns if c.endswith('_faq')])     
    long_lst = sorted([c for c in df.columns if c.endswith('_long')]) 
    cross_lst = sorted([c for c in df.columns if c.endswith('_cross')]) 
    ours_lst = sorted([c for c in df.columns if c.endswith('_')])
    
    adni_lst = sorted(list(set(list(df.columns)).difference(set(adas_lst), set(neuro_lst), set(gds_lst), set(faq_lst), 
                                                            set(long_lst), set(cross_lst), set(ours_lst))))

    
    dct = {}
    if 'adni' in pattern:
        dct[f'adni (#{len(adni_lst)})'] = adni_lst  
    if 'adas' in pattern:
        dct[f'adas (#{len(adas_lst)})'] = adas_lst       
    if 'neuro' in pattern:
        dct[f'neuro (#{len(neuro_lst)})'] = neuro_lst
    if 'gds' in pattern:
        dct[f'gds (#{len(gds_lst)})'] = gds_lst            
    if 'faq' in pattern:
        dct[f'faq (#{len(faq_lst)})'] = faq_lst         
    if 'long' in pattern:
        dct[f'long (#{len(long_lst)})'] = long_lst
    if 'cross' in pattern:
        dct[f'cross (#{len(cross_lst)})'] = cross_lst
    if '_' in pattern:
        dct[f'ours (#{len(ours_lst)})'] = ours_lst
        
    df1 = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dct.items() ]))
    df1.replace(np.NaN, '', inplace=True)
    print(f'Total number of columns: {len(df.columns)}\n')
    return df1


def package_versions(installedOnly=False, theMostImportant=[]):
    """
    Created on Wed Nov  1 13:00:07 2017
    @author: Marek

    based on: mk_package_info.py

    USAGE:
        installedOnly = True / False - if True ommits not installed package
        mostImportant = ['numpy', 'scipy', 'seaborn'] - list of interested packages only


    C: 2017.11.01
    M: 2021.03.30
    """
    import sys
    import importlib
    import platform as pl

    print("\n")
    print("Computer name: {}".format(pl.node()))
    print("Operating system: {}, {}".format(pl.system(), pl.architecture()[0]))

    print(f"\nPython path: {sys.executable}")
    print ("Python version: {}\n".format(sys.version))    


    not_installed_info = ''
    pkgs_lst = []
    vers_lst = []
    mk_packages = ['numpy', 'scipy', 'pandas', 'seaborn', 'matplotlib', 'sklearn', 'skimage', 'OpenGL', 'nibabel', 'dicom', 
                  'PyInstaller', 'PIL', 'imageio', 'cython', 'csv', 'json', 'statsmodels', 'ipywidgets', 'eli5', 'pdpbox', 'joblib', 'networkx']

    for p in mk_packages:
        try:
            module = importlib.import_module(p, package=None)
            #print(f'{p}: {module.__version__}')
            ver = module.__version__
        except:
            #print(f'{p} {not_installed_info}')
            ver = not_installed_info
        pkgs_lst.append(p)
        vers_lst.append(ver)        
    #############################      
    try:
        import tkinter
        ver = tkinter.TkVersion
    except ImportError:
        ver = not_installed_info
    pkgs_lst.append('tkinter')
    vers_lst.append(ver)
    #############################
    try:
        import PyQt5
        from PyQt5 import QtCore
        ver = QtCore.QT_VERSION_STR
    except ImportError:
        ver = not_installed_info
    pkgs_lst.append('PyQt5')
    vers_lst.append(ver)
    #############################
    try:
        import vtk
        ver = vtk.vtkVersion.GetVTKSourceVersion()
    except ImportError:
        ver = not_installed_info
    pkgs_lst.append('vtk')
    vers_lst.append(ver)
    #############################
    try:
        import itk
        ver = itk.Version.GetITKSourceVersion()
    except ImportError:
        ver = not_installed_info
    pkgs_lst.append('itk')
    vers_lst.append(ver)
    #############################
    
    
    try:
        import pandas as pd
        df = pd.DataFrame.from_dict({'module':pkgs_lst, 'version':vers_lst})
        df['module2'] = df['module'].str.lower()
        df.sort_values(by=['module2'], inplace=True)
        df.index=range(1, len(df)+1)
        df = df[['module', 'version']]

        if installedOnly:
            df = df.loc[df.version != not_installed_info]

        if len(theMostImportant):
            df = df.loc[df.module.isin(theMostImportant)]
        return df
    except ImportError:
        return print(zip(pkgs_lst, vers_lst))
