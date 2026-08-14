# Guion de presentación — EMOTIANS (Avances 1 y 2)

**Duración estimada:** 6 a 8 minutos.  
**Objetivo:** mostrar una aplicación de visión por computador para clasificación
de cuatro expresiones faciales, construida por etapas y sin modelos
preentrenados.

> Antes de presentar: abre VS Code en la raíz del proyecto y deja abierta una
> terminal en esa misma carpeta. No afirmes que la aplicación ya detecta
> emociones en vivo: esa integración es el Avance 3.

## 1. Apertura — 30 segundos

### Mostrar

Abre `README.md` y después `docs/ARCHITECTURE.md`.

### Decir

> El proyecto se llama EMOTIANS. Su propósito es reconocer cuatro expresiones
> visibles: felicidad, tristeza, enojo y sorpresa, utilizando la cámara de un
> computador. El proyecto se divide en cuatro avances para construir cada parte
> de manera verificable. Los avances que presento hoy son la base de cámara y el
> entrenamiento del clasificador.

> Una decisión importante es que no usamos modelos preentrenados: no usamos
> FER, DeepFace ni Haar cascades. La red neuronal se crea y se entrena desde
> cero con imágenes del dataset FER2013.

## 2. Arquitectura objetivo — 45 segundos

### Mostrar

En `docs/ARCHITECTURE.md`, busca la sección **Target runtime flow**.

### Decir

> El flujo final será: cámara, OpenCV, detección de rostro, preprocesamiento a
> 48 por 48 en escala de grises, CNN, probabilidades por clase y finalmente el
> resultado dibujado sobre el video.

> En este momento ya están listos la captura de cámara, el preprocesamiento y
> la CNN entrenada. El detector de rostros y la conexión de la CNN con la cámara
> se desarrollarán en el Avance 3.

## 3. Avance 1: cámara y base técnica — 1 minuto

### Mostrar

1. Abre `docs/AVANCE_1.md`.
2. Abre `src/emotians/app.py`.
3. Abre `src/emotians/vision/overlay.py`.

### Decir

> El Avance 1 implementa la base ejecutable por consola. El archivo `app.py`
> es el punto de entrada: recibe los argumentos, abre la cámara con OpenCV y
> ejecuta el ciclo frame por frame.

> En `overlay.py` se calcula y dibuja el marco amarillo central, el mensaje de
> guía y los FPS. Este marco no detecta un rostro; solamente ayuda a posicionar
> a la persona. Esto evita afirmar una funcionalidad que todavía no existe.

### Demostración opcional

En la terminal ejecuta:

```powershell
python -m emotians.app --mode preview
```

Di mientras se ve la ventana:

> Aquí se comprueba que OpenCV puede abrir la cámara, procesar cada frame y
> dibujar información visual. Se cierra con `q` o `Esc`.

## 4. Avance 2: datos y CNN — 2 minutos

### Mostrar

1. Abre `docs/AVANCE_2.md`.
2. Abre `src/emotians/data/fer2013.py`.
3. Abre `src/emotians/data/dataset.py`.
4. Abre `src/emotians/models/emotion_cnn.py`.

### Decir

> Para el segundo avance se utilizó FER2013 como fuente de imágenes, no como
> modelo. El adaptador `fer2013.py` conserva las clases angry, happy y sad, y
> convierte surprise en surprised. Las clases disgust, fear y neutral se
> descartan para mantener el alcance de cuatro emociones.

> El archivo `dataset.py` lee las imágenes con OpenCV, las lleva a 48 por 48,
> las convierte a escala de grises y normaliza sus píxeles. También incluye
> aumentos leves solo durante entrenamiento, como espejo horizontal y cambio
> pequeño de iluminación.

> El archivo `emotion_cnn.py` contiene nuestra CNN. Está compuesta por bloques
> de convolución, normalización por lotes, ReLU, pooling y una salida de cuatro
> probabilidades. Sus pesos comienzan aleatoriamente y se ajustan durante el
> entrenamiento; por eso no dependemos de un modelo externo ya entrenado.

## 5. Preparación, entrenamiento y resultados — 1 minuto 30 segundos

### Mostrar

1. Abre `data/processed/splits.csv`.
2. Abre `src/emotians/training/engine.py`.
3. Abre `artifacts/runs/stage_2/summary.json`.
4. Opcionalmente abre `artifacts/runs/stage_2/history.json`.

### Decir

> El archivo `splits.csv` registra qué imágenes se usan en entrenamiento,
> validación y prueba. Con FER2013 no hay identificadores de personas, así que
> se conserva el conjunto de prueba original y se obtiene validación
> estratificada desde el conjunto de entrenamiento. Esta es una limitación que
> está documentada y no se oculta.

> El archivo `engine.py` implementa el ciclo de entrenamiento con PyTorch,
> guarda el mejor checkpoint de validación y registra el historial. La terminal
> muestra una barra de progreso por época, pérdida acumulada y accuracy de
> validación.

> El entrenamiento ya finalizó con 20 épocas. El mejor resultado de validación
> fue **78.34% de accuracy**, alcanzado en la época 12 sobre 2,881 imágenes de
> validación. El checkpoint se guardó como `best_expression_cnn.pt`.

> Este valor es una métrica de validación, no una promesa de precisión en una
> cámara real. Antes de terminar el Avance 2 debemos ejecutar la evaluación del
> conjunto de prueba y revisar las métricas por cada emoción.

## 6. Evaluación pendiente — 30 segundos

### Mostrar

La terminal y el comando siguiente, sin ejecutarlo si no quieres esperar:

```powershell
python -m emotians.app --mode evaluate --data-root "C:\Users\Jhonny Avendaño\.cache\kagglehub\datasets\msambare\fer2013\versions\1" --checkpoint artifacts\runs\stage_2\best_expression_cnn.pt
```

### Decir

> El siguiente paso inmediato es evaluar el checkpoint en el conjunto de prueba
> que se reservó desde el inicio. El programa reportará accuracy, matriz de
> confusión y precision, recall y F1 para cada una de las cuatro emociones.

## 7. Cierre y próximos pasos — 30 segundos

### Mostrar

Vuelve a `docs/ARCHITECTURE.md`, sección **Four development stages**.

### Decir

> Los Avances 1 y 2 dejan una base comprobable: cámara funcional, datos
> preparados, una CNN propia entrenada y un checkpoint guardado. En el Avance 3
> entrenaremos un detector facial desde cero y conectaremos el recorte del
> rostro con la CNN para mostrar la etiqueta en tiempo real. El Avance 4 se
> enfocará en pruebas, errores por clase, límites y entrega final.

> Es importante describir el resultado como clasificación de expresiones
> faciales visibles, no como una medición infalible de las emociones internas
> de una persona.

## Rutas rápidas para la presentación

| Tema | Archivo que debes abrir |
| --- | --- |
| Visión general | `README.md` |
| Arquitectura y cuatro etapas | `docs/ARCHITECTURE.md` |
| Avance 1 | `docs/AVANCE_1.md` |
| Cámara y consola | `src/emotians/app.py` |
| Marco y FPS de OpenCV | `src/emotians/vision/overlay.py` |
| Avance 2 | `docs/AVANCE_2.md` |
| Adaptación de FER2013 | `src/emotians/data/fer2013.py` |
| Preprocesamiento | `src/emotians/data/dataset.py` |
| CNN propia | `src/emotians/models/emotion_cnn.py` |
| Entrenamiento y barra | `src/emotians/training/engine.py` |
| Mejor resultado | `artifacts/runs/stage_2/summary.json` |
