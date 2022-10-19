import pandas as pd 
from pydantic import BaseModel
import joblib
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import numpy as np
import gc
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from os.path import join
from typing import TypeVar
warnings.simplefilter(action='ignore', category=FutureWarning)


# 2. Class which describes a single flower measurements
class ClientData(BaseModel):
    application_train : str
    application_test  : str
    bureau : str
    bureau_balance  : str
    previous_application  : str
    POS_CASH_balance : str
    installments_payments : str
    credit_card_balance : str
    
    


# 3. Class for training the model and making predictions
class ClientModel:
    # 6. Class constructor, loads the dataset and loads the model
    #    if exists. If not, calls the _train_model method and 
    #    saves the model
    def __init__(self):
        df=pd.read_csv('df_prepro.csv')
        self.df = df
        self.model_fname_ = 'model_lgbm.pkl'
        self.model = joblib.load(self.model_fname_)
        
    def predict_target(self, data_in):
        prediction = self.model.predict(data_in)
        probability = self.model.predict_proba(data_in)
        return prediction[0], probability[0][1]
    
    
    
    
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df,ohcol,newohcol,i, nan_as_category = True, stack = True):
    df.head()
    df=df.reset_index(drop=True)
    original_columns = list(df.columns)

    categorical_columns = ohcol[i]
    new_columns = newohcol[i]
    other_col=[c for c in df.columns if c not in categorical_columns]
    df_prepro=pd.DataFrame()
    for skid in range(len(df)):
        zero_data = np.zeros(shape=(1,len(new_columns)))
        d = pd.DataFrame(zero_data, columns=new_columns)
        for col in categorical_columns:
            if df.isna()[col][skid]==True :
                if i!=0:
                    d[col+'_nan']=1
            else :
                d[col+'_'+df[col].astype(str)[skid]]=1
            
        d_other=pd.DataFrame([df[other_col].iloc[skid,:].values],columns=other_col)
        d=pd.concat([d,d_other],1)
        df_prepro=pd.concat([d,df_prepro],0)
    


            
            
        
    
    return df_prepro, categorical_columns,new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(client_data,ohcol,newohcol, nan_as_category = False):
    # Read data and merge
    df = pd.read_json(client_data['application_train'])
    test_df = pd.read_json(client_data['application_test'])
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, new_columns,cat_cols = one_hot_encoder(df,ohcol,newohcol,0, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df, new_columns,cat_cols

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(client_data,ohcol,newohcol, nan_as_category = True):
    bureau = pd.read_json(client_data['bureau'])
    bb = pd.read_json(client_data['bureau_balance'])
    
 
    bb, new_columns,bb_cat = one_hot_encoder(bb,ohcol,newohcol,1, nan_as_category)
    
    
    bureau, new_columns2,bureau_cat = one_hot_encoder(bureau,ohcol,newohcol,2,nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
 
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg ,new_columns,new_columns2,bb_cat,bureau_cat

# Preprocess previous_applications.csv
def previous_applications(client_data,ohcol,newohcol, nan_as_category = True):
    prev = pd.read_json(client_data['previous_application'])
    prev, new_columns,cat_cols = one_hot_encoder(prev,ohcol,newohcol,3, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg, new_columns,cat_cols

# Preprocess POS_CASH_balance.csv
def pos_cash(client_data,ohcol,newohcol, nan_as_category = True):
    pos = pd.read_json(client_data['POS_CASH_balance'])
    pos, new_columns,cat_cols = one_hot_encoder(pos,ohcol,newohcol,4, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg, new_columns,cat_cols
    
# Preprocess installments_payments.csv
def installments_payments(client_data,ohcol,newohcol, nan_as_category = True):
    ins = pd.read_json(client_data['installments_payments'])
    ins, new_columns,cat_cols = one_hot_encoder(ins,ohcol,newohcol,5, nan_as_category= True, stack=False)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg, new_columns,cat_cols

# Preprocess credit_card_balance.csv
def credit_card_balance(client_data,ohcol,newohcol, nan_as_category = True):
    cc = pd.read_json(client_data['credit_card_balance'])

    

    cc, new_columns,cat_cols = one_hot_encoder(cc,ohcol,newohcol,6, nan_as_category= True)
        
        
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg,cat_cols




def preprocessing(client_data : ClientData):
    
    
    
    with open('encoder.npy', 'rb') as f:
        cols=np.load(f,allow_pickle=True)
    all_new_cols=cols[2]
    ohcol=cols[0]
    newohcol=cols[1]
    
    
    
    
    
    
    df,col,new_col = application_train_test(client_data,ohcol,newohcol)
    
    bureau,col,col2,new_col,new_col2 = bureau_and_balance(client_data,ohcol,newohcol)
    
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau
    gc.collect()
    
    prev,col,new_col = previous_applications(client_data,ohcol,newohcol)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    del prev
    gc.collect()
    
    
    pos,col,new_col = pos_cash(client_data,ohcol,newohcol)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    del pos
    gc.collect()
    
    
    ins,col,new_col = installments_payments(client_data,ohcol,newohcol)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    del ins
    gc.collect()
    
    
    cc,new_col = credit_card_balance(client_data,ohcol,newohcol)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    del cc
    gc.collect()
    
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    for col in df.select_dtypes(float).columns:
        try: 
            df[col]=df[col].astype(int)
        except:
            pass
        
    mediane=pd.read_csv('mediane.csv').drop(['Unnamed: 0','index'],1)
    for col in df.columns.drop('index',1):
        df[col]=df[col].fillna(mediane[col][0])
    
    return df