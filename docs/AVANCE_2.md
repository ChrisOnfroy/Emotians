# EMOTIANS — Avance 2: datos y CNN entrenada desde cero

## Estado

**Código implementado; pendiente de ejecutar el entrenamiento.** Esta etapa ya
incluye el pipeline para preparar imágenes etiquetadas, entrenar una CNN propia
y evaluar el checkpoint resultante. No descarga, carga ni reutiliza pesos
preentrenados.

## Avance actual

| Componente | Estado | Resultado |
| --- | --- | --- |
| Descarga de FER2013 mediante KaggleHub | Implementado | Obtiene el dataset `msambare/fer2013`. |
| Adaptación de etiquetas | Implementado | Usa cuatro emociones y descarta tres. |
| Preparación de divisiones | Implementado | Genera `data/processed/splits.csv`. |
| CNN desde cero | Implementado | `EmotionCNN` recibe rostros `48x48` en gris. |
| Entrenamiento y checkpoint | Implementado | Guarda `best_expression_cnn.pt`. |
| Evaluación | Implementado | Informa accuracy, matriz de confusión y F1 por clase. |
| Modelo entrenado | Pendiente | Requiere ejecutar el entrenamiento localmente. |

Para completar este avance en tu equipo solo faltan estos pasos:

```powershell
python -m pip install -e ".[dev]"
python -m emotians.app --mode download-fer2013
python -m emotians.app --mode train --data-root "RUTA_MOSTRADA" --epochs 20 --batch-size 32
python -m emotians.app --mode evaluate --data-root "RUTA_MOSTRADA" --checkpoint artifacts/runs/stage_2/best_expression_cnn.pt
```

## Contrato del conjunto de datos

Guarda imágenes de rostros ya recortados con esta estructura exacta:

```text
data/raw/expressions/
  person_001/
    happy/      image_001.jpg
    sad/        image_002.jpg
    angry/      image_003.jpg
    surprised/  image_004.jpg
  person_002/
    ...
```

Las únicas etiquetas admitidas son `happy`, `sad`, `angry` y `surprised`.
Usa al menos tres personas y procura que cada una tenga ejemplos de las cuatro
etiquetas. No subas imágenes faciales consentidas al repositorio: `data/raw/`
está ignorado por Git.

## Lo implementado

- Escaneo y validación de estructura, etiquetas, archivos de imagen legibles y
  número de participantes.
- División reproducible 70/15/15 por participante, no por imagen. Así una
  persona nunca queda tanto en entrenamiento como en validación o prueba.
- Manifest `CSV` en `data/processed/splits.csv` con ruta, persona, etiqueta y
  partición.
- Carga de imágenes con OpenCV en escala de grises, redimensión a `48x48` y
  normalización a `[0, 1]`.
- Aumento aplicado solo al entrenamiento: espejo horizontal y cambios leves de
  iluminación.
- `EmotionCNN`, una CNN propia de cuatro clases, creada con inicialización
  aleatoria de PyTorch.
- Entrenamiento con AdamW y guardado del mejor checkpoint de validación.
- Reporte de prueba con exactitud, matriz de confusión y precision/recall/F1
  por expresión.

## Uso

Instala las nuevas dependencias desde la raíz del proyecto:

```powershell
python -m pip install -e ".[dev]"
```

Primero valida los datos y genera las particiones:

```powershell
python -m emotians.app --mode prepare-data
```

### Alternativa: FER2013 desde Kaggle

También se admite el dataset `msambare/fer2013` que propusiste. Descárgalo y
crea su manifest con:

```powershell
python -m pip install -e ".[dev]"
python -m emotians.app --mode download-fer2013
```

Si ya descargaste el dataset dentro del proyecto, no lo descargues de nuevo.
Indica la carpeta que contiene directamente `train/` y `test/`:

```powershell
python -m emotians.app --mode prepare-fer2013 --data-root "data/raw/fer2013"
```

El comando muestra el valor de `data_root` descargado por KaggleHub. Úsalo de
forma literal en los siguientes comandos, por ejemplo:

```powershell
python -m emotians.app --mode train --data-root "RUTA_MOSTRADA" --epochs 20 --batch-size 32
python -m emotians.app --mode evaluate --data-root "RUTA_MOSTRADA" --checkpoint artifacts/runs/stage_2/best_expression_cnn.pt
```

FER2013 tiene siete etiquetas. El adaptador usa `angry`, `happy`, `sad` y
`surprise` (convertida a `surprised`) y descarta `disgust`, `fear` y `neutral`.
Además, FER2013 no incluye identificadores de participantes: conserva el
conjunto `test` original y crea validación estratificada desde `train`, pero no
puede garantizar separación por persona.

Entrena desde cero; este comando nunca recibe pesos externos:

```powershell
python -m emotians.app --mode train --epochs 20 --batch-size 32 --device auto
```

El mejor modelo se guarda en:

```text
artifacts/runs/stage_2/best_expression_cnn.pt
```

Evalúalo únicamente con la partición de prueba:

```powershell
python -m emotians.app --mode evaluate --checkpoint artifacts/runs/stage_2/best_expression_cnn.pt
```

## Límites de esta etapa

La CNN espera una imagen que ya contiene un rostro. Todavía no busca rostros
en un fotograma de webcam ni dibuja una caja facial: eso corresponde al Avance
3 y deberá entrenarse igualmente desde cero.

En Windows, el lector de imágenes admite rutas Unicode, por ejemplo una carpeta
de usuario que contenga `ñ`. Por ese motivo el proyecto decodifica los bytes con
`numpy.fromfile` y `cv2.imdecode`, en lugar de depender de `cv2.imread`.

## Criterio de aceptación del Avance 2

El Avance 2 se considera finalizado cuando se haya generado un checkpoint en
`artifacts/runs/stage_2/best_expression_cnn.pt` y el comando de evaluación
produzca métricas para las cuatro clases: `happy`, `sad`, `angry` y
`surprised`. El resultado debe revisarse por clase; una accuracy global no
basta si alguna emoción tiene precision, recall o F1 muy bajos.

## Progreso durante entrenamiento

La consola muestra una barra por cada época: lotes procesados, porcentaje,
tiempo estimado y pérdida acumulada. Al terminar una época imprime la pérdida
de entrenamiento y la accuracy de validación antes de iniciar la siguiente.
