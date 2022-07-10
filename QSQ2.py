# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 20:23:09 2022

@author: Vincent
"""
import statistics
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

#Q2 (Association)
df2 = pd.read_csv('cea-salespersons-property-transaction-records-residential.csv')
df2_HDB_resale = df2[(df2['property_type']=='HDB') & (df2['transaction_type']=='RESALE')& (df2['represented']=='SELLER')]
df2_HDB_resale['transaction_date'] = pd.to_datetime(df2_HDB_resale['transaction_date']).dt.strftime('%Y')
df2_HDB_resale['salesperson_name'] = df2_HDB_resale['salesperson_name'].astype(str)
df2_HDB_resale['salesperson_name'] = df2_HDB_resale['salesperson_name'].str.replace(r"\(.*?\)", "", regex=True) #merge duplicates
df2_HDB_resale_summary = df2_HDB_resale.groupby('transaction_date')['salesperson_name'].agg(['count','nunique']).reset_index(drop=False)
print(df2_HDB_resale_summary['count']/df2_HDB_resale_summary['nunique'])
print("variance:", statistics.variance(df2_HDB_resale_summary['count']/df2_HDB_resale_summary['nunique']))

df2_Adrian = df2_HDB_resale[df2_HDB_resale['salesperson_name'] == 'ADRIAN LIM LING CHONG'] #make a freq distribution
df2_Adrian_summary = df2_Adrian.transaction_date.value_counts().reset_index(name='Frequency').rename(columns={'index':'transaction_date'})

df2_HDB_resale2 = df2_HDB_resale[['salesperson_name','transaction_date','town']]
df2_HDB_resale3 = df2_HDB_resale2.groupby(['salesperson_name','transaction_date'], as_index=False)['town'].transform(lambda x: (x.unique() + ',').sum())
basket = df2_HDB_resale3[(df2_HDB_resale3['town'].str.contains('SEMBAWANG')) & (df2_HDB_resale3['town'].str.contains('YISHUN'))]
basket.reset_index(inplace=True)
basket = pd.get_dummies(basket.town.str.split(',',expand=True))
print(df2_HDB_resale3.town.str.count('SEMBAWANG').sum()) #17035 support = 2298/43404 = 0.13795
print(df2_HDB_resale3.town.str.count('YISHUN').sum()) #34487, support = 5678/43404 = 0.27928
print(df2_HDB_resale3[(df2_HDB_resale3['town'].str.contains('SEMBAWANG')) & (df2_HDB_resale3['town'].str.contains('YISHUN'))].count().sum())
#9668 entries contain both sembawang ans yishun
#confidence(S>B) = 9668/17035 = 0.56753
#lift(S>B) = 0.56753/0.13081 = 4.3385
#min_supp = 12/43404 = 0.0002764
'''use apriori to find most visited area'''
def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1
basket_encoded = basket.applymap(hot_encode)
basket = basket_encoded

frq_items = apriori(basket, min_support = 0.04, use_colnames = True)
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())
'''
           antecedents          consequents  ...  leverage  conviction
3  (8_KALLANG/WHAMPOA)         (5_SENGKANG)  ...  0.039799    4.029717
2         (5_SENGKANG)  (8_KALLANG/WHAMPOA)  ...  0.039799    2.083870
1        (0_WOODLANDS)           (1_YISHUN)  ...  0.034936    1.332111
0           (1_YISHUN)        (0_WOODLANDS)  ...  0.034936    1.294727
	support	confidence	lift
0	0.06681836988001655	0.3604910714285714	2.0957472510952666
1	0.06681836988001655	0.38845460012026456	2.0957472510952666
2	0.04447662391394291	0.5477707006369428	9.507804548937097
3	0.04447662391394291	0.7719928186714543	9.507804548937097

'''
