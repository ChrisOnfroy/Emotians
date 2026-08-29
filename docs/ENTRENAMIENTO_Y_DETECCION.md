1# Entrenamiento, detección y flujo completo de EMOTIANS

## Problema que se quiso resolver

El problema principal era crear un sistema capaz de reconocer expresiones faciales en tiempo real a partir de la cámara de un computador. No se trataba solo de clasificar imágenes aisladas, sino de construir una solución completa que capturara video, detectara un rostro, recortara esa zona, clasificara la expresión y mostrara el resultado visualmente. Esto implicaba combinar visión por computador, aprendizaje automático y un flujo de trabajo reproducible para entrenamiento y evaluación.

Para lograrlo, se decidió construir el proyecto por etapas. Primero se preparó la base funcional con la cámara y la visualización; luego se trabajó en la preparación de datos y el entrenamiento del modelo de expresiones; finalmente se integró la detección facial y la inferencia en vivo.

## 1. Cómo se entrenó el modelo

El entrenamiento del modelo de expresiones se hizo desde cero, sin depender de un modelo preentrenado ni de librerías externas para la parte de clasificación. La idea central fue preparar un conjunto de datos con imágenes de expresiones, convertirlas a un formato consistente y entrenar una red neuronal propia para reconocer las cuatro clases objetivo: felicidad, tristeza, enojo y sorpresa.

Este proceso incluyó varias etapas. Primero se recolectaron y organizaron las imágenes; luego se separaron en entrenamiento, validación y prueba; después se normalizaron y redimensionaron; y finalmente se entrenó la red con un optimizador y una función de pérdida adecuada. El entrenamiento no se limitó a una sola pasada, sino que se repitió durante varias épocas para ajustar los pesos del modelo y mejorar su desempeño.

Durante el proceso se evaluó la red en validación para detectar si estaba aprendiendo correctamente o si necesitaba ajustes. Cada vez que el modelo mejoraba, se guardaba un checkpoint con su estado actual. Este checkpoint era importante porque permitía reutilizar el mejor modelo entrenado en la etapa de inferencia en tiempo real. En la mejor época del entrenamiento, la red alcanzó aproximadamente un 78% de accuracy en validación, lo que indicó el punto de mayor rendimiento antes de que el modelo empezara a sobreajustarse o a estabilizarse.

El modelo de expresiones se construyó como una red neuronal convolucional compacta, pensada para trabajar con rostros recortados y convertir la imagen en una confidencia por clase. Esta red era capaz de aprender las características relevantes de cada expresión y producir una probabilidad final para cada una de las categorías.

## 2. Qué fue lo que hizo PyTorch dentro del proyecto

PyTorch fue la parte central del aprendizaje automático del sistema. Es una librería que permite definir redes neuronales, entrenarlas y ejecutarlas sobre datos reales. En este proyecto, PyTorch se utilizó para construir la red `EmotionCNN`, entrenarla con las imágenes de expresiones, calcular su error, ajustar sus pesos y guardar el mejor modelo obtenido.

Su papel fue fundamental porque sin PyTorch no sería posible transformar una colección de imágenes en un modelo que aprenda patrones visuales. En otras palabras, PyTorch fue el motor del aprendizaje automático del proyecto: permitió que la red hiciera forward pass, computara pérdidas, aplicara retropropagación y optimizara sus parámetros para mejorar la clasificación.

Esto no solo se usó para entrenar, sino también para ejecutar la inferencia final. Una vez el modelo estaba entrenado, se cargaba el checkpoint y se aplicaba sobre cada rostro detectado para obtener la probabilidad de cada expresión. De esta manera, PyTorch no solo entrenó la red, sino que también la convirtió en la base de la clasificación en tiempo real.

## 3. Cómo se conectó con OpenCV

OpenCV fue la capa que permitió que el sistema funcionara con la cámara de forma práctica y visual. Se encargó de capturar los frames, procesar la imagen, detectar la zona del rostro y mostrar el resultado final en pantalla. Esta parte era esencial porque la cámara no solo debía grabar video, sino también servir como entrada del sistema de reconocimiento.

La conexión entre OpenCV y el modelo se hizo a través de un pipeline bien definido: el video entraba en el sistema, se detectaba un rostro, ese rostro era recortado, se preprocesaba y luego se enviaba a la red neuronal para obtener la expresión. Después, el resultado se dibujaba sobre la imagen original para mostrar la etiqueta y la confianza asociada.

Esto permitió que el sistema fuera completamente en tiempo real y que se pudieran observar los resultados sobre el video en pantalla, no solo como salida numérica en consola. La integración con OpenCV también permitió validar cómo se comportaba la aplicación en condiciones reales, como iluminación, cambios de ángulo y movimiento del rostro.

## 4. ¿Por qué se usó YuNet y qué ventaja tiene?

YuNet fue elegido porque es un detector facial ligero y eficiente para trabajar con video en tiempo real. Su principal función era localizar las caras dentro de cada frame de la cámara y devolver sus coordenadas para que el sistema pudiera extraer solo la zona del rostro y no analizar toda la imagen completa.

Esto es importante porque la clasificación de expresiones no se puede hacer sobre la imagen entera sin contexto. El modelo necesita trabajar únicamente sobre el rostro para capturar mejor los detalles de los ojos, la boca y la expresión. YuNet resuelve este problema detectando la cara y proporcionándole al sistema la región de interés correcta.

YuNet además resulta útil porque su uso es más robusto y rápido que intentar localizar de forma manual o a partir de una aproximación más básica. En un sistema de visión por computador, la detección facial es una etapa crítica: si no se localiza bien la cara, toda la clasificación posterior se vuelve inconsistente.

## 5. ¿Por qué no se usó la detección de rostro con el dataset de expresiones?

La razón principal fue que el dataset de expresiones no estaba diseñado para entrenar un detector facial, sino para entrenar un clasificador de expresiones. Sus imágenes están centradas en la cara, pero no contienen una representación formal de las ubicaciones exactas de los rostros ni de su localización dentro de la escena. Además, ese tipo de dataset no ofrece la misma estructura de datos que un conjunto de anotaciones para detección facial.

En otras palabras, el dataset de expresiones servía para aprender qué expresión está mostrando una cara, no para aprender dónde está esa cara en la imagen. Eso significa que usarlo para localizar rostros sería una mala decisión, porque no se cuenta con anotaciones apropiadas de cajas delimitadoras y el problema de detección requiere información distinta a la clasificación.

Por eso se optó por separar los problemas. La detección de rostro se resolvió con YuNet, que está especializado en localizar caras, mientras que el dataset de expresiones se usó exclusivamente para entrenar el modelo de expresiones. Esta separación hace que cada parte del sistema esté enfocada en su objetivo correcto y mejora la calidad de la solución global.

## 6. Qué se logró con todo este proceso

Con la combinación de datos, PyTorch, OpenCV y YuNet, el proyecto logró construir una solución completa de reconocimiento de expresiones faciales en tiempo real. Primero se preparó el conjunto de datos y el flujo de entrenamiento; luego se entrenó un modelo desde cero; después se integró la detección facial y la clasificación de expresiones; y finalmente se mostró el resultado sobre la imagen en tiempo real.

El resultado más importante no fue solo tener una red funcionando, sino tener un sistema completo, reproducible y comprendido por etapas. Eso permite entender desde el inicio qué problema se quería resolver, qué limitaciones existían, cómo se tomó la decisión técnica y qué papel jugó cada herramienta en el proceso.

## Conclusión

El proyecto nació de la necesidad de reconocer expresiones faciales a partir de una cámara, pero para lograrlo fue necesario resolver varios problemas distintos: preparación de datos, entrenamiento del modelo, integración con cámara, detección de rostros y clasificación en tiempo real. PyTorch fue la base del aprendizaje del modelo, OpenCV fue la infraestructura visual del sistema y YuNet fue la solución para detectar rostros de forma eficiente. La decisión de no usar el dataset de expresiones para detección se basó en una diferencia clara de propósito: uno sirve para clasificación y el otro para localización. Esa separación fue clave para que el sistema funcionara de manera ordenada y sólida.
