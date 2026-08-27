# Avance 3: detector de rostro e integración en tiempo real

Este avance inicia el Stage 3 descrito en `docs/ARCHITECTURE.md`: reemplazar
el ROI central por un detector de rostro entrenado desde cero y conectar ese
detector con el clasificador de expresión.

## Que quedo implementado

- Contratos estables en `src/emotians/models/contracts.py`:
  `Detection`, `Prediction`, `FrameResult`, `FaceDetector` y
  `ExpressionClassifier`.
- Detector propio `TinyFaceDetector` en `src/emotians/models/face_detector.py`.
  No carga Haar cascades, OpenCV DNN, FER, DeepFace ni pesos preentrenados.
- Dataset de anotaciones de cajas en `src/emotians/data/face_boxes.py`.
- Entrenamiento y evaluación del detector en `src/emotians/training/detector.py`
  con IoU, precisión, recall, falsos positivos y falsos negativos.
- Pipeline `detector -> crop -> expression CNN -> smoothing` en
  `src/emotians/inference/pipeline.py`.
- Overlay Stage 3 con caja roja/verde, etiqueta, confianza y FPS.
- Comandos de consola para entrenar, evaluar y correr el demo en vivo.

## Formato del manifiesto de cajas

El archivo CSV del detector debe tener estas columnas:

```text
path,split,face_present,x_min,y_min,x_max,y_max
```

Ejemplo:

```text
path,split,face_present,x_min,y_min,x_max,y_max
subject_01/frame_001.jpg,train,true,130,80,390,410
subject_01/no_face_001.jpg,train,false,,,,
subject_02/frame_010.jpg,validation,true,122,96,366,388
subject_03/frame_021.jpg,test,true,118,90,372,400
```

`path` es relativo a `--face-data-root`. Las coordenadas están en pixeles de la
imagen original y usan el orden `(x_min, y_min, x_max, y_max)`. Las filas con
`face_present=false` sirven como negativos para que el detector aprenda casos
sin rostro.

## Comandos principales

Capturar fotos propias de expresiones con la webcam:

```text
python -m emotians.app --mode collect-expressions --camera 0 --subject-id subject_01 --capture-label happy
```

Con la ventana abierta, presiona `c` para guardar una foto recortada por la guia
central. Repite cambiando `--capture-label` por `sad`, `angry` y `surprised`.

Camino rapido con FER2013 ya recortado:

```text
python -m emotians.app --mode prepare-fer2013-detector --data-root data/raw/fer2013 --manifest data/processed/splits.csv --face-manifest data/processed/face_boxes.csv
```

Entrenar el detector:

```text
python -m emotians.app --mode train-detector --face-data-root data/raw/fer2013 --face-manifest data/processed/face_boxes.csv --detector-output-dir artifacts/runs/stage_3_detector --epochs 20 --batch-size 16
```

Evaluar el detector:

```text
python -m emotians.app --mode evaluate-detector --face-data-root data/raw/fer2013 --face-manifest data/processed/face_boxes.csv --detector-checkpoint artifacts/runs/stage_3_detector/best_face_detector.pt
```

Correr el demo en vivo con detector y clasificador ya entrenados:

```text
python -m emotians.app --mode live-demo --detector-checkpoint artifacts/runs/stage_3_detector/best_face_detector.pt --expression-checkpoint artifacts/runs/stage_2/best_expression_cnn.pt
```

## Alcance actual

El detector actual predice una caja principal por frame. Esto permite reemplazar
el ROI central para un demo individual frente a webcam. El manejo completo de
multiples rostros queda como una ampliacion posterior del detector, no como una
dependencia oculta de modelos externos.

El camino rapido desde FER2013 usa cajas de imagen completa porque FER2013 ya
viene recortado alrededor del rostro. Para mejor rendimiento en webcam real,
conviene reemplazar ese manifiesto por frames anotados con la cara dentro de una
escena completa y algunos ejemplos sin rostro.

El sistema clasifica expresiones visibles dentro de las cuatro clases del
proyecto: `happy`, `sad`, `angry` y `surprised`. No debe presentarse como una
lectura confiable del estado emocional interno de una persona.
