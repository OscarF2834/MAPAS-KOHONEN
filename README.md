# 🧠 Demostración interactiva · Mapas Auto-Organizados de Kohonen (SOM)

**GRUPO 10 · "Mapas Auto-Organizados de Kohonen (SOM): Redes que Ordenan Solas"**

Demostración en Python que acompaña las diapositivas: la red SOM está escrita
**desde cero** (solo `numpy` y `matplotlib`) — BMU, vecindario gaussiano,
decaimiento de η y σ, matriz-U — para que cada paso del algoritmo sea visible
en pantalla y **totalmente interactivo**.

---

## ▶ Cómo ejecutarla

```bash
cd ~/Documentos/DEMOSTRACION\ -\ KOHOSEN
python3 demostracion_som.py
```

Requisitos (ya instalados en esta máquina):

```bash
pip install numpy matplotlib --user --break-system-packages
```

> La demo elige automáticamente el primer backend gráfico disponible
> (GTK3Agg → TkAgg → QtAgg → MacOSX). Si no encuentra ninguno, corre en modo
> "sin ventana" y solo imprime las métricas en la terminal.

---

## 🖱 Cómo se usa (¡es interactiva!)

Al arrancar aparece el **MENÚ PRINCIPAL**: una ventana con 9 botones
(uno por capítulo de la presentación) y un botón grande
**«▶ REPRODUCIR LA DEMO COMPLETA»** que lanza todos los capítulos en orden.

- Haz **clic** en un capítulo para ver solo ese.
- Cada capítulo abre **su propia ventana**: para avanzar, **cierra la
  ventana** (✕) — el menú sigue abierto por si quieres ver otro capítulo.
- **Cierra la ventana del menú** para terminar.

### Controles comunes en casi todas las ventanas

| Control | Qué hace |
|---|---|
| **⟲ repetir** (botón) o tecla **R** | vuelve a entrenar la red en vivo, con datos nuevos |
| barra de herramientas abajo | **zoom 🔍 / pan ✋** en cualquier gráfico |
| cerrar la ventana (✕) | termina ese capítulo y devuelve el control al menú |

### Interactividad de cada fase

| Botón del menú | Diapositiva | Qué puedes interactuar |
|---|---|---|
| **1 · ¿Qué es un SOM?** | Título / definición | El mapa 12×12 se entrena en vivo con colores RGB; botón ⟲ o tecla **R** re-entrena con otra semilla |
| **2 · Cuatro movimientos** | Algoritmo (4 pasos) | Cada dato llega y se ve ENTRADA→BMU→vecindario→decaimiento; botón ⟲ / **R** repite con nuevos datos |
| **3 · Dimensión & semejanza** | Reducción de dimensiones | **Haz CLIC** en un punto (izquierda) → resalta su BMU en el mapa. **Clic en una casilla** (derecha) → resalta todos los puntos que caen allí |
| **4 · Medios: noticias** | Caso real 01 | **Pasa el RATÓN (o clic)** sobre cada neurona y lee los titulares que guardó esa casilla; clic "fija" la casilla |
| **5 · Seguridad: anomalías** | Caso real 02 | **Mueve el slider k·σ** y el umbral de alarma, las X y el recuento se actualizan al instante |
| **6 · ¿Por qué un SOM?** | Ventajas | 4 mini-experimentos entrenados al momento (sin etiquetas, matriz-U, formas, ruido) |
| **7 · Exige criterio** | Limitaciones | Comparación 5×5 vs 10×10 vs 22×22 y las 4 tensiones a gestionar |
| **8 · SOM del futuro** | Futuro / tiempo real | Mapa vivo con **deriva de datos** y botón **⏸ pausa / ▶ seguir** para congelar el flujo |
| **9 · Cierre** | Conclusión | ORDENA · REVELA · INVITA — y los nombres del grupo |

### Atajos de teclado

- **R** — repetir/re-entrenar (fases 1 y 2).
- ✕ o `Ctrl+W` — cerrar cada ventana.

---

## 🎥 Consejos para presentarla

1. Abre el menú y usa **REPRODUCIR TODO** si la idea es flujo continuo; usa
   los capítulos sueltos si quieres detener el ritmo y responder preguntas.
2. En la fase 5, **pide al público un valor de k·σ** ("¿qué tan paranoicos
   queremos estar?") y muévelo con ellos.
3. En la fase 4, pasa el ratón por varias neuronas: es el momento "ajá" de
   que el mapa guarda documentos de verdad.
4. En la fase 3, haz clic en un punto de color llamativo: el público ve el
   vínculo datos crudos ↔ casilla del mapa.

---

## 🗂 Estructura del código

- `class SOM` — la red escrita desde cero:
  `paso()` (BMU + Gaussiana + regla de Kohonen), `entrenar()`, `bmu_de()`
  y `matriz_u()`.
- `fase1() … fase9()` — un capítulo por diapositiva, cada uno con su ventana.
- `men()` — el menú principal interactivo.
- Todo se dibuja con la estética de la presentación (azul, dorado, chips redondeados, texto en español).

## 🩺 Solución de problemas

| Síntoma | Solución |
|---|---|
| *tkinter no disponible* | ya está resuelto: la demo usa GTK3Agg; no requiere tkinter |
| No se abre ninguna ventana | instala un backend: `sudo apt install python3-gi gir1.2-gtk-3.0` o `pip install PyQt5 --user --break-system-packages` |
| Se ve lento | sube `REFRESCO_CADA` (línea superior del script) a 20 |
| Ventana congelada | es la ⏸ de la fase 8 — pulsa el botón ▶ seguir :-)

¡Que lo disfruten! 🎉
