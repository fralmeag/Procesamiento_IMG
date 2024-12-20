# LibreriaProImg

**LibreriaProImg** es una biblioteca para procesamiento de imágenes digitales en Python, desarrollada con el objetivo de facilitar tareas comunes como lectura, manipulación, filtrado y análisis de imágenes. Esta biblioteca está orientada a estudiantes y profesionales interesados en computación gráfica, visión por computadora y procesamiento de imágenes.

## Características principales

- **Lectura y visualización de imágenes**
  - Lectura de imágenes en formatos comunes como JPG y PNG.
  - Visualización de imágenes en RGB, escala de grises o binarizadas.

- **Conversión y manipulación**
  - Conversión de RGB a escala de grises.
  - Recorte, escalado y traslación de imágenes.

- **Análisis de imágenes**
  - Cálculo de histogramas, media y desviación estándar.
  - Umbralización utilizando el método de Otsu.

- **Operaciones aritméticas**
  - Suma, resta, multiplicación y división de imágenes.

- **Filtrado y ruido**
  - Adición de ruido (Gaussian, Salt & Pepper, Speckle).
  - Filtrado promedio y estadístico.

- **Transformaciones morfológicas**
  - Operaciones como dilatación, erosín y cerrado.

- **Transformaciones avanzadas**
  - Transformada de Hough.
  - Transformadas rápidas de Fourier.
  - Filtros Sobel, Prewitt, Gaussian y Laplacian.

## Instalación

Para usar **LibreriaProImg**, simplemente clona este repositorio y copia el archivo `LibreriaProImg_v1.py` a tu proyecto.

```bash
# Clona el repositorio
git clone https://github.com/tu_usuario/LibreriaProImg.git

# Cambia al directorio del proyecto
cd LibreriaProImg
```

Asegúrate de tener Python 3.9 o superior instalado junto con las dependencias necesarias:

```bash
pip install numpy numpy
pip install numpy matplotlib
```

## Uso

Ejemplo de lectura, conversión a escala de grises y visualización de una imagen:

```python
from LibreriaProImg_v1 import imread, rgb2gray, imshow

# Lee la imagen
imagen = imread('cartagena.jpg')

# Convierte a escala de grises
grayscale = rgb2gray(imagen)

# Muestra la imagen en escala de grises
imshow(grayscale)
```

## Documentación
Cada función está documentada dentro del archivo `LibreriaProImg_v1.py` con descripciones de sus parámetros, uso y ejemplos.

## Contribuciones

Las contribuciones son bienvenidas. Si deseas mejorar esta biblioteca o añadir nuevas funciones, sigue estos pasos:

1. Haz un fork de este repositorio.
2. Crea una nueva rama para tus cambios:
   ```bash
   git checkout -b nueva-funcion
   ```
3. Realiza tus modificaciones y realiza un commit:
   ```bash
   git commit -m "Agrega nueva función para XYZ"
   ```
4. Envía un Pull Request para su revisión.

## Autores

Esta biblioteca fue desarrollada por:
- PhD. Jimmy Alexander Cortes Osorio
- MSc. Francisco Alejandro Medina
- Ing. Juliana Gómez Osorio

Universidad Tecnológica de Pereira

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
