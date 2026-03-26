# Resumen ejecutivo del proyecto

## Nombre
Sistema de Análisis de Ventas Minoristas

## Problema
Una empresa del sector retail necesita entender cómo se comportan sus ventas por ciudad y por periodo para tomar decisiones con más criterio.

## Solución implementada
Se desarrolló un flujo de análisis en Python que:
1. consolida datos de ventas,
2. calcula métricas descriptivas clave,
3. visualiza tendencias por ciudad y año,
4. normaliza ventas para comparación,
5. entrena un modelo de regresión lineal para proyección de ventas.

## Enfoque técnico
- **Pandas** para manipulación, agregaciones y estadísticas.
- **NumPy** para normalización al rango $[0, 1]$.
- **Matplotlib** para visualización de ventas agregadas.
- **Scikit-learn** para entrenamiento y evaluación predictiva (MSE).

## Resultado
El proyecto ofrece una base analítica clara para:
- detectar sedes con mayor rendimiento,
- observar crecimiento interanual,
- simular ventas futuras por mes y año,
- justificar decisiones de inventario y marketing.

## Impacto en negocio
Este tipo de solución ayuda a reducir la dependencia de la intuición y refuerza la toma de decisiones basada en datos.

## Próximos pasos sugeridos
- ampliar el dataset a más periodos,
- robustecer validación del modelo,
- comparar algoritmos y variables adicionales,
- publicar una versión con interfaz web.
