Hiperheurística de selección adaptativa para eliminación de neblina en imágenes
Descripción

Este proyecto presenta una hiperheurística de selección adaptativa para optimizar el proceso de eliminación de neblina (image dehazing) en imágenes digitales.

La propuesta utiliza el modelo Dark Channel Prior (DCP), complementado con un filtro guiado (Guided Filter), y una estrategia hiperheurística que selecciona dinámicamente entre cuatro operadores metaheurísticos para optimizar los parámetros del proceso de restauración.

Objetivo

Desarrollar y evaluar una estrategia de optimización basada en hiperheurísticas para mejorar la eliminación de neblina en imágenes, buscando preservar los detalles, el contraste y las características visuales de las escenas originales.

Metodología

La propuesta contempla los siguientes componentes:

Modelo de eliminación de neblina: Dark Channel Prior (DCP).
Filtrado: Guided Filter.
Optimización: hiperheurística de selección adaptativa.
Operadores: búsqueda aleatoria global, mutación gaussiana, evolución diferencial e intensificación alrededor de la mejor solución.
Conjunto de datos: RESIDE-SOTS.
Métricas de evaluación: Peak Signal-to-Noise Ratio (PSNR) y Structural Similarity Index Measure (SSIM).

Resultados

Los resultados se organizan en archivos de métricas, registros del proceso de búsqueda e imágenes comparativas. La evaluación considera tanto el desempeño cuantitativo mediante PSNR y SSIM como la calidad visual de las imágenes restauradas.

Tecnologías utilizadas
Python
NumPy
OpenCV
Pandas
Matplotlib
RESIDE-SOTS
Reproducibilidad

El código fuente y los archivos de resultados se encuentran organizados en sus respectivas carpetas. Las instrucciones de instalación y ejecución se documentarán junto con el código.
