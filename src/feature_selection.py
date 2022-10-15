
# import all the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy



class CategoricalEncoding:
    def __init__(self, df, encoding_type, categorical_features,features, handle_na = False):
        self.df = df
        self.cat_feats = categorical_features
        self.features = features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = {}
        self.binary_encoders = {}
        self.ordinal_encoders = {}
        self.ohe = None

        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-99999")
        self.output_df = self.df.copy(deep= True)


    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df

    def _ordinal_encoding(self):
        for c in self.cat_feats:
            oe = OrdinalEncoder()
            oe.fit(self.df[c].values)
            self.output_df.loc[:, c] =  oe.transform(self.df[c].values)
            self.ordinal_encoders[c] = oe
        return self.output_df
    
    
    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "ordinal":
            return self._ordinal_encoding()
        else:
            raise Exception("Encoding type not understood")
    
    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:,c] = dataframe.loc[:, c].astype(str).fillna("-99999")
        return dataframe
        

    
        



