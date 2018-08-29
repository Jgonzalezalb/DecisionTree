# Imports needed for the script
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
#%matplotlib inline
#plt.rcParams['figure.figsize'] = (16, 9)
#plt.style.use('ggplot')
from sklearn import tree
from sklearn.model_selection import KFold

from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont



nba = pd.read_csv('./Data/NBA_player_of_the_week.csv')
#nba = nba.groupby('Season')

nba.head(3)

nba['Team'].value_counts()
nba['Age'].value_counts()
nba['Position'].value_counts()
nba['Weight'].value_counts()
nba['Seasons in league'].value_counts()

# Mapping:
# WEST CONFERENCE
# NORTHWEST : 0
# PACIFIC : 1
# SOUTHWEST : 2

# EAST CONFERENCE
# ATLANTIC : 3
# CENTRAL : 4
# SOUTHEAST : 5


nba['TeamEncoded'] = nba['Team'].map({'Golden State Warriors': 1,
                            'Los Angeles Lakers': 1,
                            'San Antonio Spurs': 2,
                            'Cleveland Cavaliers': 4,
                            'Miami Heat': 5,
                            'Houston Rockets': 2,
                            'Utah Jazz': 0,
                            'Phoenix Suns': 1,
                            'Chicago Bulls': 4,
                            'Orlando Magic': 5,
                            'Boston Celtics': 3,
                            'Denver Nuggets': 0,
                            'Oklahoma City Thunder': 0,
                            'Philadelphia Sixers': 3,
                            'New York Knicks': 3,
                            'Portland Trail Blazers': 0,
                            'Atlanta Hawks': 5,
                            'New Jersey Nets': 3,
                            'Toronto Raptors': 3,
                            'Dallas Mavericks': 2,
                            'Detroit Pistons': 4,
                            'Los Angeles Clippers': 1,
                            'Milwaukee Bucks': 4,
                            'Minnesota Timberwolves': 0,
                            'Indiana Pacers': 4,
                            'Washington Wizards': 5,
                            'Sacramento Kings': 1,
                            'Seattle SuperSonics': 0,
                            'Charlotte Hornets': 5,
                            'New Orleans Hornets': 2 ,
                            'Charlotte Bobcats': 5,
                            'Memphis Grizzlies': 2,
                            'Washington Bullets': 5,
                            'New Orleans Pelicans': 2 ,
                            'Brooklyn Nets': 3})

nba.loc[nba['Age'] <= 23, 'AgeEncoded'] = 0
nba.loc[(nba['Age'] > 23) & (nba['Age'] <= 26), 'AgeEncoded'] = 1
nba.loc[(nba['Age'] > 26) & (nba['Age'] <= 29), 'AgeEncoded'] = 2
nba.loc[(nba['Age'] > 29) & (nba['Age'] <= 32), 'AgeEncoded'] = 3
nba.loc[nba['Age'] > 32, 'AgeEncoded'] = 4

nba.head(4)
# SG : 0
# PG : 1
# C : 2
# PF : 3
# SF : 4

nba['PositionEncoded'] = nba['Position'].map({'G': 0,
                            'SG': 0,
                            'C': 2,
                            'PF': 3,
                            'F': 3,
                            'PG': 1,
                            'SF': 4,
                            'FC': 2,
                            'GF': 1,
                            'F-C': 2,
                            'G-F': 1})


# TODO: nba['WeightPounds'] = nba['Weight']
# la idea es pasar todo a pounds multiplicando los valores de la columna
# que estan en kg (son str) por 2,20462 y una vez limpia clasificar la columna
# en 4 o 5 categorias como la de abajo:

#nba['WeightPounds'].str.contains('kg').replace(nba['Weight'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')**2.20462)
#nba['WeightPounds']

nba.loc[nba['Seasons in league'] <= 3, 'SeasonsEncoded'] = 0
nba.loc[(nba['Seasons in league'] > 3) & (nba['Seasons in league'] <= 5), 'SeasonsEncoded'] = 1
nba.loc[(nba['Seasons in league'] > 5) & (nba['Seasons in league'] <= 7), 'SeasonsEncoded'] = 2
nba.loc[(nba['Seasons in league'] > 7) & (nba['Seasons in league'] <= 10), 'SeasonsEncoded'] = 3
nba.loc[nba['Seasons in league'] > 10, 'SeasonsEncoded'] = 4






nba['TeamEncoded'].value_counts()
nba['AgeEncoded'].value_counts()
nba['PositionEncoded'].value_counts()
nba['SeasonsEncoded'].value_counts()

# Lo mas probable para ser jugador de la semana es pertenecer a la
# división PACIFIC (1), tener entre 23 y 26 años (2), jugar en la
# posicion de SG (0) y llevar menos de tres temporadas en la liga.

nbaEncoded = nba[['TeamEncoded', 'AgeEncoded', 'PositionEncoded', 'SeasonsEncoded']]

nbaEncoded[['AgeEncoded', 'TeamEncoded']].groupby(['AgeEncoded'] , as_index=False).agg(['mean', 'count', 'sum'])
nbaEncoded[['PositionEncoded', 'TeamEncoded']].groupby(['PositionEncoded'] , as_index=False).agg(['mean', 'count', 'sum'])
nbaEncoded[['SeasonsEncoded', 'TeamEncoded']].groupby(['SeasonsEncoded'] , as_index=False).agg(['mean', 'count', 'sum'])


#  HOLA!!! -----> Estoy por aqui!

# TODO: Otro error que tenía: sb.factorplot('age', data=nba, kind="count", aspect=3)

# Se me ocurre hacer algo como esto:
# nba.loc[filtro_kg, 'WeightPounds'] = procesar_texto(nba.loc[filtro_kg,'Weight']) * 2.3

# donde filtro_kg es un mask para detectar si tiene o no el string
# y procesar_texto tiene que quitar kg de cada palabra y despues pasarlo a
# numerico (estas dos cosas se pueden hacer con un apply y una lambda)

nba['WeightPounds'] = nba['Weight']
# una parte veo que la tienes :
filtro_kg = nba['WeightPounds'].str.contains('kg') # Lo necesitas almacenar para luego

# ahora seria remplazar el kg  y pasarlo a numerico algo como Estoy

nba.loc[filtro_kg, 'WeightPounds'] = nba.loc[filtro_kg, 'WeightPounds'].replace({r'([0-9]*)(kg)': r'\1'}, regex=True)
# Prueba ahora!
nba.loc[filtro_kg, 'WeightPounds'] = nba.loc[filtro_kg,'WeightPounds'].astype(float )

# ahora ya puedes multiplicar

nba.loc[filtro_kg, 'WeightPounds'] = nba.loc[filtro_kg, 'WeightPounds']  * 2.3







cv = KFold(n_splits=10) # Numero deseado de "folds" que haremos
accuracies = list()
max_attributes = len(list(nbaEncoded))
depth_range = range(1, max_attributes + 1)

# Testearemos la profundidad de 1 a cantidad de atributos +1
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(criterion='entropy',
                                             min_samples_split=20,
                                             min_samples_leaf=5,
                                             max_depth = depth,
                                             class_weight={1:3.5})
    for train_fold, valid_fold in cv.split(nbaEncoded):
        f_train = nbaEncoded.loc[train_fold]
        f_valid = nbaEncoded.loc[valid_fold]

        model = tree_model.fit(X = f_train.drop(['TeamEncoded'], axis=1),
                               y = f_train["TeamEncoded"])
        valid_acc = model.score(X = f_valid.drop(['TeamEncoded'], axis=1),
                                y = f_valid["TeamEncoded"]) # calculamos la precision con el segmento de validacion
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)

# Mostramos los resultados obtenidos
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))





# Crear arrays de entrenamiento y las etiquetas que indican si llegó a top o no
y_train = nbaEncoded['TeamEncoded']
x_train = nbaEncoded.drop(['TeamEncoded'], axis=1).values

# Crear Arbol de decision con profundidad = 4
decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
                                            min_samples_split=20,
                                            min_samples_leaf=5,
                                            max_depth = 4,
                                            class_weight={1:3.5})
decision_tree.fit(x_train, y_train)

# exportar el modelo a archivo .dot
with open(r"tree1.dot", 'w') as f:
     f = tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 7,
                              impurity = True,
                              feature_names = list(nbaEncoded.drop(['TeamEncoded'], axis=1)),
                              #class_names = ['No', 'MVP'],
                              rounded = True,
                              filled= True )

# Convertir el archivo .dot a png para poder visualizarlo
check_call(['dot','-Tpng',r'tree1.dot','-o',r'tree1.png'])
PImage("tree1.png")




acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
print(acc_decision_tree)
