#Chargement des librairies

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import datetime
import pandas as pd
import numpy as np
import datetime
from datetime import date
import plotly.express as px
#Financial functions for python
import ffn

#Tableau de bord
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#Chargement des donnees du S&P_500
data = pd.read_csv('Donnees.csv',parse_dates=["Date"], index_col="Date")
indice = pd.read_csv('indice.csv',parse_dates=["Date"], index_col="Date")
fichier = pd.read_csv('Info.csv')

#Nettoyage des valeurs infinies et minimes
data = data.fillna(method='ffill')
indice = indice.fillna(method='ffill')

#CALCUL DES INDICATEURS

#Rendements historiques en log
LogReturns = np.log(data/data.shift(1))
SpReturn = np.log(indice/indice.shift(1))

#Rendements moyen journalier
daily_mean = LogReturns.mean() * 252

#Volatilité moyen journalière
daily_vol = LogReturns.std() * np.sqrt(252)

#Ratio de sharpe journalier
daily_sharpe = daily_mean / daily_vol

#Drawdown sur les prix
maxDD = ffn.core.calc_max_drawdown(data)

#Moyenne mobile
MA30 = data.rolling(window=30).mean()
MA250 = data.rolling(window=250).mean()

#Calcul du rendement sur toute la periode
def rendement_total(prices):
    return (prices.iloc[-1]/prices.iloc[1]) -1

total_return = rendement_total(data)


#NOUVEAU DATASET AVEC NOS INDICATEURS
dataset = pd.DataFrame({'Rendements_moyen':daily_mean,
                        'Volatilite':daily_vol,'Sharpe':daily_sharpe,
                        'maxDrawDown':maxDD,'Rendement_total':total_return})

#REGRESSION LINEAIRE 

#fig = px.scatter(data_frame=None, x=LogReturns['GOOG'], y=SpReturn,
                 #trendline="ols",trendline_color_override='red',
                 #width= 600, height=600)
#fig.show()

#Résultats du modèle
#results = px.get_trendline_results(fig)
#results.px_fit_results.iloc[0].summary()

#ANALYSES EN COMPOSANTES PRINCIPALES

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#On normalise nos donnees dataset
scaler = StandardScaler()
scaler.fit(dataset)
x = scaler.transform(dataset)

pca = PCA(n_components=2)
x[np.isnan(x)] = np.median(x[~np.isnan(x)])
pca.fit(x)
x_pca = pca.transform(x)
principalDf = pd.DataFrame(data = x_pca)
principalDf = principalDf.join(pd.DataFrame({'Stock':data.columns.values}))

#On ajoute le secteur de chaque actif pour une meilleur visualisation 
sector1=[]
for i in range(len(principalDf['Stock'])):
    for j in range(len(fichier['ticker'])):
        if principalDf['Stock'][i] == fichier['ticker'][j]:
            sector1.append(fichier['Industry'][j])

principalDf = principalDf.join(pd.DataFrame({'Industry':sector1}))

#fig = px.scatter(principalDf, x=0,
                 #y=1, color=principalDf['Industry'])
#fig.show()

#Clustering sur donnees financieres : Apprentissage non supervise

#Preparation des données
dataset = dataset.set_index(data.columns.values)

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

X = dataset[['Rendements_moyen','Volatilite','Sharpe','maxDrawDown','Rendement_total']]
s_imputer = SimpleImputer(missing_values = np.nan, strategy='mean',verbose=0)
s_imputer = s_imputer.fit(X)
X = s_imputer.transform(X)
X = pd.DataFrame(X)

#On veut definir le bon nombre de cluster à appliquer sur nos données

distorsions = []
for k in range(2, 20):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    distorsions.append(k_means.inertia_)

#px.line(data_frame=None, x=range(2, 20), y=distorsions, title="Methode de Elbow", width=600, height=600)

#Modele
kmeans = KMeans(n_clusters=5)

#On applique notre modele sur nos donnees
kmeans.fit(X)

#Statistiques du modele

#SSE
#print("Kmeans inertie :", kmeans.inertia_)


#fig = px.scatter(data_frame=None, x= X.iloc[:,0], y= X.iloc[:,1], color=kmeans.predict(X),
           #width=600,height=600, title="Classification par K-means clustering")
#fig.show()

dataset['Cluster']=kmeans.labels_

app.layout = html.Div(
        children=[
                html.Div(
                        children=[
                                html.P(children="📈", className="header-emoji"),
                                html.H1(
                                        children="Analyses des actifs du S&P 500",className="header-title"
                                ),
                                html.P(
                                        children="Utilisation des methodes de machines learning"
                                        " pour analyser les characteristiques des actifs financiers",
                                        className="header-description",
                                ),
                        ],
                        className="header",
                ),
                html.Div([
                        html.Div([
                                html.H6(children='Daily mean',
                                        style={'textAlign':'center',
                                                'color':'black'}),
                                html.P(f"{str(round(daily_mean['AAPL']*100,2)) + ' %':}",
                                        style={'textAlign' :'center',
                                                'color':'green',
                                                'fontSize':30}),
                                html.P('Rendement annuel 2020 : ' + f"{str(round(np.prod(LogReturns['AAPL'] + 1) ** (252/LogReturns['AAPL'].shape[0]) - 1,2)*100) + ' %':}",
                                        style={'textAlign' : 'center',
                                                'color' : 'green',
                                                'fontSize':15,
                                                'margin-top':'-18px'})

                        ], className='card_container three columns'),

                        html.Div([
                                html.H6(children='Daily volatility',
                                        style={'textAlign':'center',
                                                'color':'black'}),
                                html.P(f"{str(round(daily_vol['AAPL']*100,2)) + ' %':}",
                                        style={'textAlign' :'center',
                                                'color':'red',
                                                'fontSize':30}),
                                html.P('Volatilite annuel 2020 : ' + f"{str(round(daily_vol['AAPL']*np.sqrt(250))) + ' %':}",
                                        style={'textAlign' : 'center',
                                                'color' : 'red',
                                                'fontSize':15,
                                                'margin-top':'-18px'}),
                        ], className = 'card_container three columns'),

                        html.Div([
                                html.H6(children='Ratio de Sharpe',
                                        style={'textAlign':'center',
                                                'color':'black'}),
                                html.P(f"{str(round(daily_sharpe['AAPL'],2)):}",
                                        style={'textAlign' :'center',
                                                'color':'orange',
                                                'fontSize':30}),
                                html.P('Sharpe ratio annuel 2020 : ' + f"{str(round(daily_sharpe['AAPL']*12,2)):}",
                                        style={'textAlign' : 'center',
                                                'color' : 'orange',
                                                'fontSize':15,
                                                'margin-top':'-18px'}),
                        ], className = 'card_container three columns'),

                        html.Div([
                                html.H6(children='Max Draw Down',
                                        style={'textAlign':'center',
                                                'color':'black'}),
                                html.P(f"{str(round(maxDD['AAPL'],2)):}",
                                        style={'textAlign' :'center',
                                                'color':'#e55467',
                                                'fontSize':30}),
                                html.P('Over the last three years, since 2018',
                                        style={'textAlign' : 'center',
                                                'color' : '#e55467',
                                                'fontSize':15,
                                                'margin-top':'-18px'}),
                        ], className = 'card_container three columns'),

                ], className='row flex display'),
        html.Div([
                html.Div([
                        html.P('Select a stock', className='fix_label', style={'color': 'black'}),
                        dcc.Dropdown(id='t_ticker',
                                        multi=False,
                                        searchable=True,
                                        value='AAPL',
                                        placeholder='Select a ticker',
                                        options=[{'label': value, 'value' : value}
                                        for value in (LogReturns.columns)],className='dcc.compon'),
                        html.P('Mis a jour le: ' + ' ' + str(LogReturns.index[-1]),
                                className='fix_label', style={'text-align':'center', 'color':'green'}),
                        dcc.Graph(id='price', config={'displayModeBar':False}, className='dcc_compon',
                        style={'margin-top':'20px'}),
                        dcc.Graph(id='returns', config={'displayModeBar':False}, className='dcc_compon',
                        style={'margin-top':'20px'}),

                ], className='create_container three columns'),
                html.Div([
                        dcc.Graph(id='reg',figure={},config={'displayModeBar':False})
                ],className='create_container eight and a half columns')

        ], className = 'row flex display')
        ]
)

@app.callback(Output('price','figure'),
                [Input('t_ticker','value')])

def update_return(t_ticker):
        last_price = round(data[t_ticker].iloc[-1],2)
        last_price2 = round(data[t_ticker].iloc[-2],2)
        return {
                'data': [go.Indicator(
                        mode='number+delta',
                        value = last_price,
                        delta = {'reference':last_price2,
                                'position' : 'right',
                                'valueformat' : ',2%',
                                'relative': False,
                                'font':{'size':10}},
                        number={'valueformat': ',g',
                                'font' : {'size':15}},
                        domain={'y': [0,1], 'x': [0,1]}
                )],
                'layout': go.Layout(
                        title = {'text': 'Last price',
                                'y':1,
                                'x':0.5,
                                'xanchor':'center',
                                'yanchor':'top'},
                        height=50,
                )
        }

@app.callback(Output('returns','figure'),
                [Input('t_ticker','value')])

def update_return(t_ticker):
        last_return = round(LogReturns[t_ticker].iloc[-1],2)
        last_return2 = round(LogReturns[t_ticker].iloc[-2],2)
        return {
                'data': [go.Indicator(
                        mode='number+delta',
                        value = last_return,
                        delta = {'reference':last_return2,
                                'position' : 'right',
                                'valueformat' : ',2%',
                                'relative': False,
                                'font':{'size':10}},
                        number={'valueformat': '.g',
                                'font' : {'size':15}},
                        domain={'y': [0,1], 'x': [0,1]}
                )],
                'layout': go.Layout(
                        title = {'text': 'Last return',
                                'y':1,
                                'x':0.5,
                                'xanchor':'center',
                                'yanchor':'top'},
                        height=50,
                )
        }


@app.callback(Output('reg','figure'),
                [Input('t_ticker','value')])
def update_reg(t_ticker): 
        fig = px.line(x=data.index,y=data[t_ticker])
        fig.add_traces(go.Scatter(
                name="30 Moving average",
                mode="lines",
                x = data.index,
                y = MA30[t_ticker]
        ))
        fig.add_traces(go.Scatter(
                name="250 Moving average",
                mode="lines",
                x=data.index,
                y=MA250[t_ticker]
        ))
        return fig


if __name__== '__main__':
  app.run_server(debug=True)

