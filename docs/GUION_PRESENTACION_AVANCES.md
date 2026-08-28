# Guion (versión hablada y tranquila) — EMOTIANS (Avances 1 y 2)

Duración estimada: 6–8 minutos.

Nota importante: no voy a entrar a leer ni explicar el código línea por línea.
No he repasado cada archivo en detalle; presentaré la idea, los resultados y los
pasos. Si surge una pregunta técnica puntual, pediré apoyo del equipo o lo
anotaré para revisarlo después. Esto evita confusiones durante la presentación.

1) Apertura (30s)
- Mostrar: `README.md`, `docs/ARCHITECTURE.md`.
- Qué decir, en voz natural:
	"Hola, soy [tu nombre]. Les presento EMOTIANS: una pequeña aplicación que
	reconoce cuatro expresiones visibles — feliz, triste, enojado y sorprendido —
	usando la cámara del computador. Hoy muestro los dos primeros avances: la
	base de la cámara y el entrenamiento del clasificador. No uso modelos
	preentrenados; la red se entrena aquí con FER2013."

2) Arquitectura objetivo (45s)
- Mostrar: `docs/ARCHITECTURE.md` (sección Target runtime flow).
- Qué decir, relajado:
	"La idea final es sencilla: la cámara captura el video, detectamos el
	rostro, recortamos y preprocesamos a 48×48 en escala de grises, la CNN
	predice probabilidades por cada clase y mostramos la etiqueta sobre el
	video. Hoy ya tenemos la captura, el preprocesamiento y la CNN entrenada; el
	detector en vivo será parte del Avance 3."

3) Avance 1 — cámara y base técnica (1 min)
- Mostrar: `docs/AVANCE_1.md`, `src/emotians/app.py`, `src/emotians/vision/overlay.py`.
- Qué decir y cómo mostrarlo:
	"El primer avance es la base ejecutable: abrir la cámara y procesar frames
	con OpenCV. En la demo de consola verán un marco guía y los FPS; ese marco
	solo ayuda a posicionarse, no hace detección de rostros todavía."
- Demo corta (opcional):
	```bash
	python -m emotians.app --mode preview
	```
	Mientras se ve la ventana decir: "Esto comprueba que la cámara funciona y
	que podemos procesar frames en tiempo real. Se cierra con `q` o `Esc`."

4) Avance 2 — datos y CNN (2 min)
- Mostrar: `docs/AVANCE_2.md`, `src/emotians/data/fer2013.py`,
	`src/emotians/data/dataset.py`, `src/emotians/models/emotion_cnn.py`.
- Qué decir, sencillo:
	"En este avance usamos FER2013 como fuente de imágenes: adaptamos las
	clases para quedarnos con cuatro emociones y descartamos otras que no
	forman parte del alcance. El pipeline convierte las imágenes a 48×48 en
	escala de grises, normaliza y aplica aumentos ligeros durante el
	entrenamiento (como espejado o cambios leves de iluminación)."

	"La CNN es una red pequeña con capas de convolución, batch-norm y pooling;
	se entrena desde cero y guarda el mejor checkpoint."

5) Entrenamiento y resultados (1:30)
- Mostrar: `data/processed/splits.csv`, `src/emotians/training/engine.py`,
	`artifacts/runs/stage_2/summary.json`.
- Qué decir, claro y honesto:
	"Guardamos los splits de entrenamiento/validación/prueba. El proceso de
	entrenamiento con PyTorch registró historial y guardó el mejor checkpoint.
	En validación obtuvimos aproximadamente 78% de accuracy (mejor época).
	Esto es una métrica interna: no significa que tenga esa misma precisión en
	la cámara real sin más validación."

6) Evaluación pendiente (30s)
- Mostrar la intención de evaluar y cómo hacerlo (no es necesario ejecutar).
- Qué decir:
	"El siguiente paso es evaluar el checkpoint en el conjunto de prueba para
	obtener la matriz de confusión y métricas por clase (precision, recall,
	F1). Eso ayudará a entender qué emociones se confunden más."

7) Cierre y próximos pasos (30s)
- Mostrar: `docs/ARCHITECTURE.md` (Four development stages).
- Qué decir, directo:
	"Los Avances 1 y 2 dejan una base: cámara, datos procesados, una CNN
	entrenada y un checkpoint. En el Avance 3 entrenaremos un detector facial
	y conectaremos el pipeline para mostrar las etiquetas en vivo. El Avance 4
	será sobre pruebas, análisis por clase y ajustes finales."

Tips para presentar (frases cortas y naturales)
- "No voy a explicar el código línea a línea; presento la idea y las pruebas."
- "Si quieren detalles técnicos, tomo la pregunta y lo revisamos con el
	equipo después."
- Mantén las explicaciones en 1–2 frases y muestra ejemplos o archivos cuando
	sea útil.

Atajos (qué abrir durante la demo)
- `README.md` — visión general
- `docs/ARCHITECTURE.md` — flujo objetivo
- `docs/AVANCE_1.md` — demo de cámara
- `src/emotians/app.py` — punto de entrada
- `src/emotians/vision/overlay.py` — marco y FPS
- `docs/AVANCE_2.md` — datos y entrenamiento
- `src/emotians/models/emotion_cnn.py` — la red
- `artifacts/runs/stage_2/summary.json` — resultado resumido

Fin.
