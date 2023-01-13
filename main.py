import time
from estadistica import Estadistica
from data_limpia import Data
import os

def main():
    start = time.time()
    print(" ")
    print(" ")    
    print("El programa comenzo a ejecutarse a las:")
    print("{} (Local time)".format(time.ctime(start)))
    print(" ")

    #######################################################################
    #######################################################################
    # main program here:
    directorio_actual = os.path.realpath(os.path.dirname(__file__))
    os.chdir(directorio_actual)

    print("Escribe el nombre del archivo a usar como 'data'")
    input_file = input()
    datos = Data(input_file)
    dataframe = datos.crea_df_csv()
    dataframe_limpio = datos.limpia_datos(dataframe)
    dataframe_sin_outliers = datos.quitar_outliers(dataframe_limpio)
    datos_finales = datos.exportar(dataframe_sin_outliers)
    
    df_bajo_estudio = Estadistica(datos_finales[1])
    df_bajo_estudio.tipos_de_datos()
    df_bajo_estudio.normalidad()



    #######################################################################
    #######################################################################

    end = time.time()
    total_time = end - start
    print("{} (Local time)".format(time.ctime(start)))
    print(" ")
    print("Tiempo total de ejecucion:")
    print("\n"+ str(total_time))
    print(" ")
    print(" ")


# if __name__ == 'main':
main()