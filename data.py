#Chargement des librairies
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import date
import bs4 as bs
import requests


#Chargement des donnees du S&P_500
resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table',{'class':'wikitable sortable'})
tickers = []
names = []
Sector = []
for row in table.findAll('tr')[1:]:
  ticker = row.findAll('td')[0].text
  name = row.findAll('td')[1].text
  secteur = row.findAll('td')[3].text
  tickers.append(ticker)
  names.append(name)
  Sector.append(secteur)

tickers = list(map(lambda s: s.strip(), tickers))
names = list(map(lambda s: s.strip(), names))
Sector = list(map(lambda s: s.strip(), Sector))

#Fichier avec toutes les informations qu'on veut
tickerdf = pd.DataFrame(tickers,columns=['ticker'])
namesdf = pd.DataFrame(names,columns=['Nom'])
Sectordf = pd.DataFrame(Sector,columns=['Industry'])

fichier = pd.concat([tickerdf, namesdf, Sectordf], axis=1)

start= datetime.datetime(2018,1,1)
data = yf.download(tickers ,start=start, period='1d')
start2 = datetime.datetime(2017,12,28)
#On conserve uniquement le prix ajuste et remplace les valeurs manquantes par 0
data = data['Adj Close']
indice = yf.download('^GSPC' ,start=start2, period='1d')
indice = indice['Adj Close']

data= pd.DataFrame(data, index=data.index)
indice = pd.DataFrame(indice, index=data.index)
fichier = pd.DataFrame(fichier)

data.to_csv('Donnees.csv',header=True, index=True)
indice.to_csv('indice.csv',header=True, index=True)
fichier.to_csv('Info.csv',header=True, index=True)

import requests

demo= '8c2b607f6ba5ce388df2a8d15b59237b'

Financial_ratio = {}

for item in tickers:
    try:
        BS = requests.get(f'https://fmpcloud.io/api/v3/ratios/{item}?limit=40&apikey={demo}')
        BS = BS.json()
        
        #Chargement des ratio 
        PE_ratio = BS[0]['priceEarningsRatio']
        Price_to_book = BS[0]['priceEarningsRatio']
        Debt_equity = BS[0]['debtEquityRatio']
        FCF = BS[0]['freeCashFlowPerShare']
        PE_togrowth = BS[0]['priceEarningsToGrowthRatio']
        
        #Chargement dans le dictionnaire des ratios pour chaque actif
        Financial_ratio[item] = {}
        Financial_ratio[item]['PE_ratio'] = PE_ratio
        Financial_ratio[item]['Price_to_book'] = Price_to_book
        Financial_ratio[item]['Debt_equity'] = Debt_equity
        Financial_ratio[item]['FCF'] = FCF
        Financial_ratio[item]['PE_togrowth'] = PE_togrowth
    except:
        pass 

data2 = pd.DataFrame.from_dict(Financial_ratio, orient='index')
data2.to_csv('Donnees2.csv',header=True, index=True)
