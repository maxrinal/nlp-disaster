import matplotlib.pyplot as plt
import pandas as pd
df_test = pd.read_csv('data/train.csv')

print ("---------------df_test_info---------------")
df_test.info()

print ("---------------df_test_target_value_counts---------------")
df_test.target.value_counts()

print ("---------------df_test_head---------------")
df_test.head(10)

df_test['text_len']  = df_test.text.str.len()


# Genero un nuevo dataframe para poner las dos columnas
df = pd.DataFrame()

# Separo cada una de las columnas
df_text_len_false = df_test.loc[ df_test.target == 0 , ['text_len']  ]
df_text_len_true  = df_test.loc[ df_test.target == 1 , ['text_len']  ]

df_text_len_false = df_text_len_false.reset_index(drop=True)
df_text_len_true  = df_text_len_true.reset_index(drop=True)

df['Target 0'] = df_text_len_false.text_len
df['Target 1'] = df_text_len_true.text_len

df.describe()

"""Comparo el histograma de ocurrencias de los tweets verdaderos contra los tweets falsos.
Debo normalizar los mismos, ya que no tengo la misma cantidad de ocurrencias de cada uno de los tipos.
Puede observarse que ambos tienen distribuciones muy similares, superpuestas y no existe un horizonte que las delimite  por lo cual no puede utilizarse la longitud del tweet como método de clasificación
"""

fig, axes = plt.subplots(nrows=1, ncols=1)

ax_hist = df.plot.hist(ax = axes , bins= 20, density=1, alpha=0.4 , grid=True, title = 'Histograma normalizado con longitud de tweets por clase')
ax_hist.set_xlabel('Longitud del tweet')
ax_hist.set_ylabel('Densidad de probabilidad estimada')


#~ ax_boxplot = df.boxplot(ax = axes[1] , vert=False )
#~ ax_boxplot.set_title( 'Media,Varianza,Min,Max')
#~ ax_boxplot.set_xlabel('Longitud del tweet')
#~ ax_boxplot.set_xlabel('Probabilidad')

plt.subplots_adjust(hspace=0.5)

fig.savefig("visualizacion.png", bbox_inches='tight')
# Si quiero visualizarlo
plt.show()


