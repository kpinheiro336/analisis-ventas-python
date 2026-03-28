# Sistema de Análisis de Ventas Minoristas

Proyecto en Python centrado en el análisis de ventas de varias tiendas minoristas. A partir de un conjunto de datos ficticio, se estudia la evolución de las ventas por ciudad y año, se normalizan los datos y se entrena un modelo sencillo de regresión lineal para estimar ventas futuras.

## Objetivo

El objetivo del proyecto es mostrar un flujo básico de trabajo en análisis de datos aplicado a negocio:
- análisis descriptivo de ventas,
- visualización de tendencias por ciudad y año,
- normalización de datos con NumPy,
- predicción de ventas con Scikit-learn.

## Contexto del caso

El proyecto parte de un escenario del sector retail en el que una empresa quiere entender mejor su rendimiento comercial. Para ello, se trabaja con datos ficticios de ventas de Madrid, Barcelona y Valencia durante 2021 y 2022, con desglose mensual y por categoría.

## Tecnologías

- Python 3.10+
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Estructura del proyecto

- `Analisis_Ventas.py`: script principal con menú interactivo, estadísticas, visualizaciones y predicción.
- `requirements.txt`: dependencias necesarias para ejecutar el proyecto.
- `docs/RESUMEN.md`: resumen breve del proyecto orientado a portfolio.
- `docs/ESTRUCTURA.md`: explicación de la organización del repositorio.
- `CHANGELOG.md`: registro de cambios principales.
- `AUTHORS.md`: autoría y datos de contacto.

## Instalación

1. Clona el repositorio.
2. Crea y activa un entorno virtual.
3. Instala dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python Analisis_Ventas.py
```

El script abre un menú interactivo con:
1. Dashboard de indicadores
2. Gráfico de tendencias
3. Normalización de ventas
4. Simulador predictivo
5. Conclusiones de negocio

## Flujo analítico implementado

### 1) Manipulación de datos con Pandas
- Creación del DataFrame con ventas por ciudad, año, mes y categoría.
- Cálculo de estadísticas descriptivas.
- Agrupación por ciudad y año para obtener las ventas totales.

### 2) Visualización con Matplotlib
- Gráfico de barras de ventas agregadas por ciudad y año.

### 3) Preprocesamiento con NumPy
- Normalización de ventas al rango $[0, 1]$ para facilitar comparación y modelado.

### 4) Predicción con Scikit-learn
- Variables predictoras: `Año` y `Mes_Num`.
- División entrenamiento/prueba: 80% / 20%.
- Modelo: `LinearRegression`.
- Métrica de evaluación: MSE (Error Cuadrático Medio).

## Resultados esperados

- Detectar qué ciudades aportan más ingresos.
- Observar la evolución de las ventas entre un año y otro.
- Generar proyecciones mensuales como apoyo a la planificación comercial.

## Valor para negocio

Este análisis puede servir como apoyo para:
- anticipar la demanda por mes,
- planificar inventario,
- ajustar campañas de marketing,
- detectar oportunidades de mejora por ubicación.

## Hoja de ruta (mejoras futuras)

- Incorporar más años y estacionalidad real.
- Comparar modelos (Random Forest, XGBoost, etc.).
- Añadir métricas MAE y $R^2$.
- Exportar reportes automáticos en CSV/PDF.
- Crear interfaz web (Streamlit) para uso no técnico.

## Evidencias visuales

### Menú principal

![Menú principal](assets/menu_inicial.png)

### Submenú de proyecciones y análisis

![Submenú de proyecciones y análisis](assets/submenu_proyecciones_analisis.png)

### Gráfico de ventas

![Gráfico de ventas](assets/grafico_ventas.png)

### Predicción del modelo

![Predicción del modelo](assets/prediccion_modelo.png)
