import numpy as np
import pandas as pd
import time
import os

class Data:

    """
    Crea a partir de unos datos un dataframe
    Se abrevia dataframe a veces como 'df'

    :param data: set de datos 

    :metodo crea_csv: crea un DataFrame si los datos son .csv
    :metodo drop_column: elimina las columnas de datos deseadas
    """

    def __init__(self, data):
        self.data = data
        
    def crea_df_csv(self, separacion=',', cabecera=0):

        """
        Esta funcion pretende limpiar el set de datos

        :param data: separado a priori por comas

        :return dataframe: con separacion y cabecera estandars
        """

        df = pd.read_csv(self.data, sep=separacion, header=cabecera)

        return df;

    def limpia_datos(self, df, ceros=False):

        """
        Esta funcion pretende limpiar el set de datos.
        Reemplazamos NaN por zeros y despues eliminamos los zeros generales.

        :param dataframe: usando pandas

        :return dataframe: limpio

        """
        
        if ceros == True: # si los ceros deben quitarse
            numero_total_nan = df.isna().sum()
            df_sin_nan = df.fillna(0)
            numero_total_ceros = (df == 0.00).sum()
            df_sin_ceros_sin_nan = df_sin_nan[(df_sin_nan.T != 0).any()]
            df_limpio = df_sin_ceros_sin_nan
            print(" ")
            print("Han sido eliminados:\n{} [NaN's]\n \
                    {} [ceros]".format(numero_total_nan, numero_total_ceros))
        else: # quitamos las filas con NaN
            numero_total_nan = df.isna().sum()
            df_limpio = df.dropna()
            print("Han sido eliminados:\n{} NaN's".format(numero_total_nan))

        return df_limpio;

    def quitar_outliers(self, df_limpio, intercuartilico=True, standar=False, ml=False):

        """
        Esta funcion pretende limpiar el set de datos de outliers.
        Nos basamos en el rango intercuartilico.

        :param dataframe: usando pandas
        :param intercuartilico: flag
        :param standar: flag
        :param ml: flag

        :return dataframe: limpio

        """

        if intercuartilico == True:
            Q1 = df_limpio.quantile(q=0.25)
            Q3 = df_limpio.quantile(q=0.75)
            IQR = Q3 - Q1
            df_sin_outliers = df_limpio[~((df_limpio < (Q1 - 1.5 * IQR)) |(df_limpio > (Q3 + 1.5 * IQR))).any(axis=1)]
        elif standar == True:
            #df_sin_outliers
            pass
        elif ml == True:
            #df_sin_outliers
            pass
        else:
            #df_sin_outliers
            pass

        return df_sin_outliers;

    def exportar(self, dataframe, separacion_option=False):
        nombre_archivo_listo= ""
        archivo_generado = []
        directorio_trabajo = os.getcwd()
        if self.data[-4:] == '.csv' and separacion_option == False:
            nombre_archivo_listo += '{}_limpio_listo.csv'.format(self.data[:-4])
            archivo_generado = dataframe.to_csv(nombre_archivo_listo, sep=',' ,encoding='utf-8')
            print('El archivo se encuentra en {}, con nombre:\n {}'.format(directorio_trabajo,nombre_archivo_listo))
        else:
            print("No se ha podido exportar")
            
        return archivo_generado, nombre_archivo_listo
        
# def main():
#     start = time.time()
#     print(" ")
#     print(" ")    
#     print("El programa comenzo a ejecutarse a las:")
#     print("{} (Local time)".format(time.ctime(start)))
#     print(" ")

#     #######################################################################
#     #######################################################################
#     # main program here:

#     # print("Escribe el nombre del archivo a usar como 'data'")
#     # input_file = input()
#     # datos = Data(input_file)
#     # dataframe = datos.crea_df_csv()
#     # dataframe_limpio = datos.limpia_datos(dataframe)
#     # dataframe_sin_outliers = datos.quitar_outliers(dataframe_limpio)
#     # fichero_a_estudiar = datos.exportar(dataframe_sin_outliers)

#     #######################################################################
#     #######################################################################

#     end = time.time()
#     total_time = end - start
#     print("{} (Local time)".format(time.ctime(start)))
#     print(" ")
#     print("Tiempo total de ejecucion:")
#     print("\n"+ str(total_time))
#     print(" ")
#     print(" ")


# if __name__ == 'main':

#     main()
    
#     print(" ")
#     print("El script fue importado")
# else:
#     print(" ")
#     print("El script se ejecuto directamente")

#     main()
