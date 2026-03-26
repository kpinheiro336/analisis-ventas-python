import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ------- Crear un DataFrame ------------------------------- #
data = {
    "Tiendas": ["Madrid", "Barcelona", "Valencia", "Madrid", "Barcelona", "Valencia",
               "Madrid", "Barcelona", "Valencia", "Madrid", "Barcelona", "Valencia"],
    "Año": [2021]*6 + [2022]*6,
    "Mes": ["Enero","Enero","Enero", "Febrero", "Febrero", "Febrero",
            "Enero", "Enero", "Enero","Febrero", "Febrero", "Febrero"],
    "Ventas": [1500, 2000, 1800, 1700, 2100, 1900,
               1600, 2200, 2000, 1800, 2300, 2100],
    "Categoría":["Electrónica", "Ropa", "Electrónica", "Hogar", "Ropa", "Hogar",
                 "Electrónica", "Ropa", "Hogar", "Electrónica", "Ropa", "Hogar"]
}
df = pd.DataFrame(data)

# ------- Preprocesamiento base ------------------------------- #

meses_map = {"Enero":1, "Febrero":2, "Marzo":3, "Abril":4, "Mayo":5, "Junio":6,
             "Julio":7, "Agosto":8, "Septiembre":9, "Octubre":10, "Noviembre":11, "Diciembre":12}

df['Mes_Num'] = df['Mes'].map(meses_map)
ventas_tiendas_año = df.groupby(['Tiendas', 'Año']).agg({'Ventas': 'sum'}).unstack()

# Variables globales para conectar NumPy al Modelo de predicción
v_min_global = df['Ventas'].min()
v_max_global = df['Ventas'].max()

# ---------------- FUNCIONES ---------------- #

def mostrar_estadisticas():
    print("\n--- PANEL DE CONTROL DE INDICADORES CLAVE ---")

    resumen = df['Ventas'].describe()[['mean', '50%', 'std', 'max', 'min']]
    resumen.index = ['Promedio', 'Mediana (50%)', 'Desviación Estándar', 'Venta Máxima', 'Venta Mínima']
    print("\n[1] RESUMEN GENERAL DE VENTAS:\n")
    print("\nMuestra el comportamiento global de los ingresos.\n")
    print(resumen.map(lambda x: f"{x:.2f} €").to_string()) # Formatear a 2 decimales

    stats_tiendas = df.groupby('Tiendas')['Ventas'].agg(['sum', 'mean', 'std']).sort_values(by='sum', ascending=False)
    stats_tiendas.columns = ['Ventas Totales', 'Promedio por Venta', 'Variabilidad (Std)']
    print("\n[2] RENDIMIENTO POR TIENDA (Sede de ciudad):")
    print("\nCompara qué ciudades generan más ingresos y cuáles son más constantes.\n")
    print(stats_tiendas.map(lambda x: f"{x:,.2f} €"))
   

    stats_cat = df.groupby('Categoría')['Ventas'].agg(['sum', 'count', 'mean'])
    stats_cat.columns = ['Ingresos Totales', 'Cantidad Vendida', 'Ticket Promedio']
    print("\n[3] ÉXITO POR CATEGORÍA DE PRODUCTO:")
    print("Identifica qué tipo de productos dominan el mercado.")
    stats_cat_visual = stats_cat.copy()
    stats_cat_visual['Ingresos Totales'] = stats_cat_visual['Ingresos Totales'].map(lambda x: f"{x:,.2f} €")
    stats_cat_visual['Ticket Promedio'] = stats_cat_visual['Ticket Promedio'].map(lambda x: f"{x:,.2f} €")
    print(stats_cat_visual.to_string())

    stats_año = df.groupby('Año')['Ventas'].sum()
    print("\n[4] CRECIMIENTO ANUAL:")
    print("Muestra la evolución total del negocio año tras año.")
    stats_año_euro = stats_año.map(lambda x: f"{x:,.2f} €")
    print(stats_año_euro.to_string())
    
    print("\n" + "="*40)


def mostrar_grafico():
    print("\nGENERANDO MAPA VISUAL DE RENDIMIENTO INTERANUAL...")
    ventas_tiendas_año.plot(kind='bar')
    plt.title('Ventas Totales por Tiendas(Ciudad) y Año')
    plt.xlabel('Tiendas (Ubicación Ciudad)')
    plt.ylabel('Ventas')
    plt.legend(title='Año')
    plt.show()


def preprocesamiento_numpy():
    global v_min_global, v_max_global
    print("\n--- CONSULTA DE RENDIMIENTO RELATIVO (NumPy) ---")
    print("\nEsta opción asigna una 'puntuación' de 0 a 1 a cada venta:\n")
    print("- 1.00: Representa el récord histórico de ventas.")
    print("- 0.00: Representa el punto más bajo registrado.")
    
    ventas_totales = df['Ventas'].values # Array de NumPy

    v_min_global = np.min(ventas_totales)
    v_max_global = np.max(ventas_totales)

    df['Ventas_Norm'] = (ventas_totales - v_min_global) / (v_max_global - v_min_global) # Normalización a [0,1]

    print(f"\nReferencias actuales:")
    print(f"-> Récord (1.00): {v_max_global}")
    print(f"-> Mínimo (0.00): {v_min_global}")
    
    print("\nÚltimos registros analizados (Índice de Performance):\n")
    print(df[['Tiendas', 'Mes', 'Ventas', 'Ventas_Norm']].to_string(index=False))



def modelo_prediccion():
    print("\n--- MODELO DE PREDICCIÓN ---")
    if 'Ventas_Norm' not in df.columns: # Si el usuario no pasó por la opción 3, se asegura de tener la columna normalizada para el modelo
        preprocesamiento_numpy()

    X = df[['Año', 'Mes_Num']].values
    y = df['Ventas_Norm'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    while True:
        print("\n--- MENÚ DE PROYECCIONES Y ANÁLISIS PREDICTIVO ---")
        print("1. Ver métricas del modelo")
        print("2. Ver comparación Real vs Predicho")
        print("3. Predecir ventas manualmente")
        print("0. Volver al menú principal")

        opcion = input("Selecciona una opción: ").strip()

        if opcion == "1":
            y_pred = modelo.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            print("\nCoeficientes:", modelo.coef_)
            print("Intercepto:", modelo.intercept_)
            print("Error Cuadrático Medio (MSE):", mse)

        elif opcion == "2":
            y_pred = modelo.predict(X_test)
            y_test_real = (y_test * (v_max_global - v_min_global) + v_min_global).flatten()
            y_pred_real = (y_pred * (v_max_global - v_min_global) + v_min_global).flatten()


            comparacion = pd.DataFrame({
                "Año": X_test[:, 0].astype(int),
                "Mes_ID": X_test[:, 1].astype(int),
                "Venta Real": y_test_real.round(2),
                "Venta Predicha": y_pred_real.round(2),
            })

            print("\nComparación Real vs Predicho:")
            print(f"\n{comparacion.to_string(index=False)}\n")

        elif opcion == "3":
            try:
                while True:  
                    año_input = input("Introduce el año a proyectar (ej: 2023): ").strip()
                    año = int(año_input)

                    if año > 2100:
                        print(f"\n⚠ El año {año} es demasiado lejano para una proyección fiable.")
                        print("Por favor, introduce un año máximo hasta el 2100.\n")
                    elif año < 2000:
                        print(f"\n⚠ El año {año} no es válido. Introduce un año entre 2000 y 2100.\n")
                    else:
                        break

                print(f"\nGenerando simulación estratégica para el año {año}...")

                resultados = []

                for mes, num in meses_map.items():
                    pred_norm = modelo.predict([[año, num]])[0]
                    pred_real = pred_norm * (v_max_global - v_min_global) + v_min_global # Convertir la predicción de NumPy a valor real
                    resultados.append((mes, round(pred_real, 2)))

                    

                tabla = pd.DataFrame(resultados, columns=["Mes", "Ventas Predichas"])

                print(f"\nPredicciones para {año}:\n", tabla.to_string(index=False))

                # Gráfico de la sumulación
                ver_grafico = input("\n¿Desea generar el gráfico de esta proyección? (s/n): ").lower().strip()
                
                if ver_grafico == 's':
                    plt.figure(figsize=(10, 5))
                    plt.plot(tabla["Mes"], tabla["Ventas Predichas"], marker='o', linestyle='-', color='g')
                    plt.title(f'Simulación de Tendencia de Ventas - Año {año}')
                    plt.xlabel('Meses')
                    plt.ylabel('Ventas Proyectadas (€)')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.show()

                

            except ValueError:
                print("Error: Debes introducir valores numéricos válidos.")

            except Exception as e:
                print(f"Error inesperado: {e}")

        elif opcion == "0":
            break

        else:
            print("⚠ Opción no válida.")


def conclusiones():
    print("\n--- CONCLUSIONES ---")
    print("""
1. Barcelona y Valencia destacan en ventas, especialmente en 2022.
2. Existe crecimiento de un año a otro.
3. La normalización permite comparar datos fácilmente.
4. El modelo predice con un error bajo/moderado.
5. Puede ayudar a decisiones como:
   - Gestión de inventario
   - Estrategias de marketing
   - Identificación de meses fuertes
""")
    
# ---------------- MENÚ ESTILO PROFESIONAL ---------------- #

OPCIONES_MENU = [
    ("1", "Dashboard de Performance Comercial", mostrar_estadisticas),
    ("2", "Monitor de Tendencias Visuales", mostrar_grafico),
    ("3", "Índice de Eficiencia Operativa por Tienda", preprocesamiento_numpy),
    ("4", "Simulador de Proyecciones y Metas", modelo_prediccion),
    ("5", "Informe de Insights y Estrategia Comercial", conclusiones),
    ("0", "Salir", None),
]


def mostrar_menu():
    print("\n=========== MENÚ ===========")
    for clave, descripcion, _ in OPCIONES_MENU:
        print(f"{clave}. {descripcion}")


def ejecutar_menu():
    despacho = {clave: fn for clave, _, fn in OPCIONES_MENU if fn is not None}

    while True:
        try:
            mostrar_menu()
            opcion = input("Elige una opción: ").strip()

            if opcion == "0":
                print("\nSaliendo del programa...\n")
                print("\n¡Gracias por usar el sistema de análisis de ventas! Hasta luego.\n")
                break

            funcion = despacho.get(opcion)

            if funcion:
                funcion()
            else:
                print("Opción no válida. Intenta de nuevo.")

        except Exception as e:
            print(f"Error inesperado: {e}")


# ---------------- EJECUCIÓN ---------------- #

def main():
    print("\n--- SISTEMA DE ANÁLISIS DE VENTAS ---")
    ejecutar_menu()    

if __name__ == "__main__":
    main()    


