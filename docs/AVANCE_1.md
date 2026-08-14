# EMOTIANS — Avance 1: base técnica y vista de cámara

## Estado

**Completado.** Este avance prepara la aplicación para trabajar desde consola
con la cámara del portátil. Aún no detecta rostros ni clasifica expresiones;
esa separación es intencional para construir el proyecto por etapas.

## Lo implementado

- Estructura de paquete Python bajo `src/emotians`.
- Comando de consola: `python -m emotians.app`.
- Configuración editable de cámara y vista previa en `configs/preview.yaml`.
- Apertura de la cámara con OpenCV (`cv2.VideoCapture`).
- Marco amarillo centrado que guía al usuario para posicionar el rostro.
- Medición y visualización de FPS.
- Mensaje de estado: `Collecting data — center your face in the guide`.
- Cierre seguro de cámara y ventana al presionar `q` o `Esc`.
- Validación de la configuración YAML, para informar errores de manera clara.
- Registro de eventos en consola (`logging`).
- Pruebas unitarias para la configuración y el dibujo del marco, sin necesitar
  una cámara física.
- Configuración de empaquetado en `pyproject.toml` y dependencias de Stage 1.

## Archivos principales

| Archivo | Responsabilidad |
| --- | --- |
| `src/emotians/app.py` | Punto de entrada y ciclo de webcam. |
| `src/emotians/config/settings.py` | Lee y valida el archivo YAML. |
| `src/emotians/vision/overlay.py` | Calcula el marco y dibuja textos con OpenCV. |
| `configs/preview.yaml` | Índice, resolución y apariencia de la cámara. |
| `tests/` | Pruebas sin cámara. |

## Cómo ejecutarlo

Desde la raíz del repositorio (la carpeta que contiene `pyproject.toml`):

```powershell
python -m pip install -e ".[dev]"
python -m emotians.app --config configs/preview.yaml --mode preview
```

Si la cámara no es la predeterminada, prueba otro índice:

```powershell
python -m emotians.app --camera 1 --mode preview
```

Para ejecutar las pruebas:

```powershell
python -m pytest
```

## Qué todavía no existe

No hay todavía una red neuronal, detector facial, clasificación de emociones,
ni pesos descargados. Tampoco se usan FER, DeepFace, Haar cascades ni cualquier
otro modelo preentrenado. El marco amarillo es una guía de posición, no una
detección facial.

## Avances pendientes

### Avance 2 — Datos y clasificador de expresiones

Preparar y validar el conjunto de datos para `happy`, `sad`, `angry` y
`surprised`; construir y entrenar desde cero una CNN que clasifique el recorte
del rostro; guardar resultados, métricas y checkpoints.

### Avance 3 — Detector facial e integración en tiempo real

Entrenar desde cero un detector de rostros con anotaciones de cajas; sustituir
el marco guía por una caja sobre el rostro real; conectar detector, CNN de
expresiones, suavizado temporal y etiqueta con confianza.

### Avance 4 — Evaluación y entrega

Probar con vídeos consentidos y escenarios difíciles (sin rostro, poca luz,
múltiples rostros); revisar errores por expresión; documentar límites,
reproducibilidad y pasos de demostración.

## Criterio de aceptación del Avance 1

El avance está correcto cuando la aplicación abre la cámara, muestra el marco
amarillo y los FPS, y se cierra sin dejar la cámara bloqueada con `q` o `Esc`.
