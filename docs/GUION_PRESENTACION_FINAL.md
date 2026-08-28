# Guion final de presentación — EMOTIANS

Duración sugerida: 8–10 minutos (puedes adaptar a 5–7 para pitch corto).

Intro breve (20–30s)
- "Hola, soy [tu nombre]. Les presento EMOTIANS, un prototipo para
  reconocimiento de expresiones faciales visibles usando la webcam. El proyecto
  integra captura, preparación de datos, entrenamiento de una CNN desde cero
  y una API web para demos en el navegador. No usaré el código línea por
  línea durante la demo — presentaré diseño, resultados y decisiones. Si
  quieren detalles técnicos, lo revisamos luego con el equipo."

Arquitectura en una frase (30s)
- "El flujo es: cámara → detector de rostro → recorte y preprocesamiento a
  48×48 en escala de grises → CNN (4 clases) → suavizado de probabilidades →
  resultado. También hay una API web que recibe frames JPEG y responde JSON
  con la predicción y métricas auxiliares."

Captura y experiencia de usuario (1 min)
- Mostrar: `README.md` y una demo rápida de `python -m emotians.app --mode preview`.
- Narrativa:
  "La captura se hace con OpenCV; la ventana muestra un marco guía y FPS. El
  marco ayuda a posicionarse antes de guardar ejemplos o iniciar una demo en
  vivo. Existe un modo para capturar fotos etiquetadas (presionas `c`) y crear
  un conjunto propio de entrenamiento."

Datos y preparación (1 min)
- Mostrar: `docs/AVANCE_2.md` y `data/processed/splits.csv` (si quieres).
- Narrativa:
  "Usamos FER2013 o fotos propias como fuente. El adaptador conserva cuatro
  etiquetas: `happy`, `sad`, `angry`, `surprised`. Generamos particiones
  reproducibles por persona (70/15/15) para evitar que la misma persona aparezca
  en entrenamiento y prueba. Las imágenes se normalizan a 48×48 y se aplican
  aumentos leves en entrenamiento."

Modelo y entrenamiento (1:30)
- Mostrar: `src/emotians/models/emotion_cnn.py` y `artifacts/runs/stage_2/summary.json`.
- Narrativa:
  "La CNN es pequeña y está inicializada desde cero. Se entrena con AdamW y
  usa pesos de clase para compensar desbalances. El entrenamiento guarda el
  mejor checkpoint y produce métricas por clase; en validación alcanzó ~78%.
  Esto es indicativo: hay que revisar la matriz de confusión para entender
  qué clases se confunden y si hace falta más datos o augmentos."

Pipeline de inferencia (1 min)
- Mostrar: `src/emotians/inference/pipeline.py`.
- Narrativa:
  "Para inferencia combinamos un detector de rostros (YuNet o un detector
  pequeño propio) con el clasificador. El pipeline elige la detección con
  mayor confianza, recorta la cara, clasifica y aplica un suavizado exponencial
  a las probabilidades para evitar cambios bruscos en la etiqueta. El estado
  final puede ser `ok`, `no-face`, `invalid-crop` o `low-confidence`."

Web demo y robustez (1 min)
- Mostrar: `emotians-web` (página) y explicar: cliente captura frames, envía al
  endpoint `/api/analyze` y pinta la respuesta en la UI.
- Narrativa:
  "La web funciona enviando JPEGs periódicos al servidor. Hemos añadido una
  comprobación de baja iluminación: si la imagen está demasiado oscura el API
  devuelve `status: low-light` y la UI muestra un aviso, evitando predicciones
  poco fiables. Para demos en entornos con recursos limitados, reducir la
  frecuencia o el tamaño del frame mejora la experiencia."

Validación y tests (45s)
- Mostrar: `tests/` y citar que la suite de tests pasa (si así fue).
- Narrativa:
  "Hay tests unitarios e integración ligera sobre el pipeline y la API.
  Durante el desarrollo la suite local pasó — esto ayuda a evitar regresiones
  al cambiar preprocesamiento o el servidor."

Despliegue y consideraciones de recursos (45s)
- Narrativa:
  "Para demos locales recomendamos al menos 1 vCPU y 2 GB RAM. En plataformas
  con 0.1 vCPU y 500 MB (p. ej. ciertas instancias de Render) las predicciones
  pueden ser lentas o fallar. Opciones:
  - Opción A (rápida): reducir frecuencia y tamaño de frames en el cliente.
  - Opción B (robusta): exportar el modelo a ONNX y servir con ONNX Runtime
    (posible cuantización) o aumentar recursos."

Limitaciones y ética (30s)
- Narrativa:
  "EMOTIANS clasifica expresiones faciales visibles; no mide estados emocionales
  internos. Hay limitaciones por iluminación, ángulo, diversidad de datos y
  sesgos del dataset. En cualquier uso real se deben considerar consentimiento,
  privacidad y validación específica por población objetivo."

Demo sugerida (30–60s)
- Pasos rápidos para la demo en vivo:
  1. Abrir terminal en la raíz del repo.
  2. Activar entorno: `source .venv/bin/activate`.
  3. Ejecutar: `python -m emotians.app --mode live-demo --face-model artifacts/models/face_detection_yunet_2023mar.onnx --expression-checkpoint artifacts/runs/stage_2/best_expression_cnn.pt`.
  4. Alternativa web: `emotians-web` y abrir `http://127.0.0.1:8000`.

Preguntas que puedes recibir (short answers)
- "¿Funciona en móviles?": requiere adaptar captura y detector; la web necesita
  HTTPS y permisos de cámara.
- "¿Qué pasa con la iluminación?": si está muy baja evitamos inferir y
  pedimos aumentar luz; mejoras incluyen CLAHE o corrección gamma antes de
  clasificar.
- "¿Se puede optimizar?": sí—exportar a ONNX y usar ONNX Runtime o cuantizar.

Cierre (20–30s)
- Mensaje final breve:
  "EMOTIANS es una prueba de concepto completa: captura, preparación de datos,
  entrenamiento desde cero, inferencia y demo web. Está pensado como base para
  investigación y prototipado, y admite mejoras de optimización y más datos
  para producción. Gracias — ahora respondo preguntas."

Archivo creado: `docs/GUION_PRESENTACION_FINAL.md` — puedes editar el nombre o
pedirme una versión aún más corta para usar en una diapositiva.
