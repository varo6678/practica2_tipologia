import numpy as np
import pandas as pd
import time
import os
from scipy import stats
import pingouin as pg

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import matplotlib
import matplotlib.pyplot as plt


class Estadistica:

    """
    Crea a partir de unos datos un dataframe
    Se abrevia dataframe a veces como 'df'

    :param data: set de datos 

    :metodo crea_csv: crea un DataFrame si los datos son .csv
    :metodo drop_column: elimina las columnas de datos deseadas

    support doc:
    https://pbpython.com/pandas_dtypes.html
    """

    def __init__(self, datos):
        self.datos = datos
        self.df = self._crea_df_csv(self.datos)
        self.info = self.df.info
        self.atributos = self.df.columns

    @staticmethod
    def _crea_df_csv(datos, separacion=',', cabecera=0):

        """
        Esta funcion pretende limpiar el set de datos

        :param data: separado a priori por comas

        :return dataframe: con separacion y cabecera estandars
        """

        df = pd.read_csv(datos, sep=separacion, header=cabecera)

        return df;

    def tipos_de_datos(self):
        print("Los atributos son un total de {} y son:\n".format(len(self.df.columns)))
        for atributo, tipo in zip(self.df.columns,self.df.dtypes):
            print("{} -> {}".format(atributo,tipo))


    def descripcion_atributos(self):
        
        placeholder = 0.

        return placeholder

    def normalidad(self, dagostino=True):
        lista_normales = []
        lista_no_normales = []

        print("Si la variable no es normal se indicara con un asterico '*'")

        if dagostino == True:
            # D'Agostino's K-squared test
            
            for atributo in self.atributos:
                k2, p_value = stats.normaltest(self.df[atributo])
                if p_value > 0.05:
                    print("Atributo: {} -> Estadistico = {}, p-value = {}".format(atributo, k2, p_value))
                    lista_normales.append(atributo)
                else:
                    print("Atributo: {} -> Estadistico = {}, p-value = {}*".format(atributo, k2, p_value))
                    lista_no_normales.append(atributo)

            print("Summary:")
            print("Atributos normales")
            for atributo in lista_normales:
                print(atributo)
            print("Atributos no normales")
            for atributo in lista_no_normales:
                print(atributo)

        else:
            # Kolmogorov test
            placeholder_kolmogorov = 0.
        
        return lista_normales, lista_no_normales

    def homocedasticidad(self, x, y, normalidad=True):

        print("Si la variable no es homocedastica se indicara con un asterico '*'")

        if normalidad == True:
            print("Se empleara el Bartlett test al existir normalidad")
            # Bartlett test
            # ==============================================================================
            k2, p_value = stats.bartlett(self.df[x], self.df[y])
            if p_value > 0.05:
                print("Atributos: {} y {} -> Estadistico = {}, p-value = {}".format(x, y, k2, p_value))
            else:
                print("Atributos: {} y {} -> Estadistico = {}, p-value = {}*".format(x, y, k2, p_value))
                print("No son homocedasticas") 

        else:
            print("Se empleara el Levene test al no existir normalidad")
            # Levene test
            # ==============================================================================
            k2, p_value = stats.levene(self.df[x], self.df[y])
            if p_value > 0.05:
                print("Atributos: {} y {} -> Estadistico = {}, p-value = {}".format(x, y, k2, p_value))
            else:
                print("Atributos: {} y {} -> Estadistico = {}, p-value = {}*".format(x, y, k2, p_value))
                print("No son homocedasticas")       

            # # Fligner test
            # # ==============================================================================
            # fligner_test = stats.fligner(peso_hombres, peso_mujeres, center='median')
            # fligner_test


    def correlacion(self,x,y, parametrico=False):

        if parametrico == True:
            r, p = stats.pearsonr(self.df[x], self.df[y])
            print("Atributos: {} y {} -> correlacion  de Pearson = {}, p-value = {}".format(x, y, r, p))

        else:
            r, p = stats.spearmanr(self.df[x], self.df[y])
            print("Atributos: {} y {} -> correlacion de Spearman= {}, p-value = {}".format(x, y, r, p))
            
    
    def t_test(self,x,y, correccion=False):

        if correccion == False:
            test_result = pg.ttest(x, y, alternative='two-sided', correction=False)

        else:
            test_result = pg.ttest(x, y, alternative='two-sided', correction=True)
        
        if test_result['p-val'] >= 0.05:
            print("No hay diferencia de medias")
        else:
            print("Hay diferencia de medias")


    def regresion_lineal(self, x, y):

        # https://www.cienciadedatos.net/documentos/py10-regresion-lineal-python.html

        X = self.df[[x]]
        y = self.df[y]

        X_train, X_test, y_train, y_test = train_test_split(
                                                X.values.reshape(-1,1),
                                                y.values.reshape(-1,1),
                                                train_size   = 0.8,
                                                random_state = 64,
                                                shuffle      = True,
                                            )

        modelo = LinearRegression()
        modelo.fit(X = X_train.reshape(-1, 1), y = y_train)
        

        print("Intercept:", modelo.intercept_)
        print("Coeficiente:", list(zip(X.columns, modelo.coef_.flatten(), )))
        print("Coeficiente de determinación R^2:", modelo.score(X, y))

        predicciones = modelo.predict(X = X_test)
        print(predicciones[0:3,])

        rmse = mean_squared_error(
                y_true  = y_test,
                y_pred  = predicciones,
                squared = False
            )
        print("")
        print(f"El error (rmse) de test es: {rmse}")

        # Predicciones con intervalo de confianza del 95%
        #######################################################################################################

        predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)
        predicciones['x'] = X_train[:, 1]
        predicciones['y'] = y_train
        predicciones = predicciones.sort_values('x')

        # Gráfico del modelo
        #######################################################################################################
        fig, ax = plt.subplots(figsize=(6, 3.84))

        ax.scatter(predicciones['x'], predicciones['y'], marker='o', color = "gray")
        ax.plot(predicciones['x'], predicciones["mean"], linestyle='-', label="OLS")
        ax.plot(predicciones['x'], predicciones["mean_ci_lower"], linestyle='--', color='red', label="95% CI")
        ax.plot(predicciones['x'], predicciones["mean_ci_upper"], linestyle='--', color='red')
        ax.fill_between(predicciones['x'], predicciones["mean_ci_lower"], predicciones["mean_ci_upper"], alpha=0.1)
        ax.legend();


    # def grafica_2d(self, x, y):
    #     plt.plot(x,y)
    #     plt.show()
