from explainerdashboard import ExplainerDashboard, ClassifierExplainer, RegressionExplainer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from explainerdashboard.custom import *
from dash_bootstrap_components.themes import FLATLY

import dash_bootstrap_components as dbc



#Importing Libraries & Packages
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from pandas import DataFrame

#import portalocker
from dash import Dash, callback, html, dcc, dash_table, Input, Output, State, MATCH, ALL



    
#Import the Diabetes Dataset

data=pd.read_csv('data_latest.csv')

#values.astype('float32')

#data=data.dropna()
#data = data[data['OBB']!='']

# Drop data
X= data.drop(['food_inf','core','inflation'], axis = 1)






y = data['inflation']




X.head()

#y=pd.DataFrame(data.target,columns=["target"])
y.head()

########## Dataset Split ########



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


model = RandomForestRegressor(n_estimators = 400,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                            max_depth=5,
                           random_state = 42)



model.fit(X_train, y_train.values.ravel())


#X_train, y_train, X_test, y_test = titanic_survive()
#train_names, test_names = titanic_names()
#model = RandomForestClassifier(n_estimators=50, max_depth=5)
#model.fit(X_train, y_train)




explainer = RegressionExplainer(model, X_test, y_test)




class CustomModelTab(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="Importance")
        self.importance =  ImportancesComposite(explainer,
                                title='Impact',
                                hide_importances=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.importance.layout(),
                    html.H3(f"Clearly {self.explainer.columns_ranked_by_shap()[0]} was the most important"
                            f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                            f" and {self.explainer.columns_ranked_by_shap()[2]}.")
                ])
            ])
        ])




class CustomModelTab1(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="Statistics")
        self.Reg_summary = RegressionModelStatsComposite(explainer,
                                title='Impact',
                                hide_predsvsactual=False, hide_residuals=False,
                                hide_regvscol=False)
        self.register_components()

    def layout(self):
           return dbc.Container([
               dbc.Row([
                   dbc.Col([
                       self.Reg_summary.layout(),
                      
                      
                   ])
               ])
           ])
    

class CustomPredictionsTab(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="Predictions")

        self.prediction = IndividualPredictionsComposite(explainer,
                                                    hide_predindexselector=False, hide_predictionsummary=False,
                                                    hide_contributiongraph=False, hide_pdp=False,
                                                    hide_contributiontable=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Individual Prediction:"),
                    self.prediction.layout()
                ])
                
            ])
        ])
    


class CustomPredictionsTab2(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="what if")

        self.what_if = WhatIfComposite(explainer,
                                                    hide_whatifindexselector=False, hide_inputeditor=False,
                                                    hide_whatifcontribution=False, hide_whatifpdp=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Individual Prediction:"),
                    self.what_if.layout()
                ])
                
            ])
        ])
    
    
    
    

class CustomPredictionsTab3(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="SHAP Dependencies")

        self.shap_depend = ShapDependenceComposite(explainer,
                                                    hide_shapsummary=False, hide_shapdependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("SHAP Dependencies:"),
                    self.shap_depend.layout()
                ])
                
            ])
        ])
    
    
    
class CustomPredictionsTab4(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="Decision Trees")

        self.decision_trees = DecisionTreesComposite(explainer,
                                                    hide_treeindexselector=False, hide_treesgraph=False,
                                                    hide_treepathtable=False, hide_treepathgraph=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Decision Trees:"),
                    self.decision_trees.layout()
                ])
                
            ])
        ])    
    

from dash_bootstrap_components.themes import CYBORG,PULSE,DARKLY
#from dash_bootstrap_components.themes import LUMEN



db=ExplainerDashboard(explainer, [CustomModelTab, CustomModelTab1, CustomPredictionsTab,
                               CustomPredictionsTab2, CustomPredictionsTab3, CustomPredictionsTab4], 
                        title='Macroeconomic Indicator Prediction for Nigeria', header_hide_selector=False,
                        bootstrap=CYBORG)

db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)

from explainerdashboard import ExplainerDashboard
db = ExplainerDashboard.from_config("dashboard.yaml") 
app = db.flask_server()
