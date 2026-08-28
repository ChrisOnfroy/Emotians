# Emotians

Reconocimiento de expresiones faciales en tiempo real desde la webcam.

Emotians usa OpenCV para captura y visualización, y entrena sus modelos desde
cero. El prototipo incluye comandos para previsualización local, captura de
imágenes etiquetadas, entrenamiento y una API web que recibe frames JPEG.

Resumen rápido
- Probar localmente (preview): `python -m emotians.app --config configs/preview.yaml --mode preview`
- Ejecutar la web: `emotians-web` (instalado por el package)
- Correr tests: `python -m pytest`

Instalación y entorno
1. Crear y activar un virtualenv (Python 3.10+):
```bash
python -m venv .venv
source .venv/bin/activate
```
2. Instalar dependencias de desarrollo:
```bash
python -m pip install -e ".[dev]"
```

Comandos útiles
- Previsualización (sin detector ni modelo):
	`python -m emotians.app --config configs/preview.yaml --mode preview`
- Capturar expresiones (presiona `c` para guardar):
	`python -m emotians.app --mode collect-expressions --subject-id subject_01 --capture-label happy`
- Demo en vivo con detector YuNet y checkpoint de expresión:
	`python -m emotians.app --mode live-demo --face-model artifacts/models/face_detection_yunet_2023mar.onnx --expression-checkpoint artifacts/runs/stage_2_balanced/best_expression_cnn.pt`
- Web app (después de instalar el paquete):
	`emotians-web` → abrir http://127.0.0.1:8000

Estructura importante
- `src/emotians/` — código fuente (CLI, web, inference, training, utils)
- `configs/` — archivos YAML de configuración (p. ej. `preview.yaml`)
- `data/` — imágenes crudas y datasets procesados (ignorados por git)
- `artifacts/` — modelos y checkpoints (no versionados)
- `tests/` — tests unitarios y de integración

Notas operativas
- Si la aplicación va a desplegarse en un entorno con pocos recursos (p. ej.
	0.1 vCPU / 500 MB RAM), es probable que las predicciones sean lentas o fallen.
	Para demos locales bastan 1 vCPU y 2 GB RAM; para producción recomiendamos
	más memoria o servir modelos optimizados (ONNX/quantización).
- Ya existe una comprobación de baja iluminación: si la cámara está muy oscura
	la API devuelve `status: "low-light"` y la UI muestra un aviso.

Dónde seguir
- Documentación técnica y diseño: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Informes de avance y guías de entrenamiento: [docs/AVANCE_1.md](docs/AVANCE_1.md), [docs/AVANCE_2.md](docs/AVANCE_2.md), [docs/AVANCE_3.md](docs/AVANCE_3.md)

Contacto
- Autor: Cristian Morales y Juan Benavides

