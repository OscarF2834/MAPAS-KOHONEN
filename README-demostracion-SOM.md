# 🔢 Ejercicio Numérico · Mapas Auto-Organizados de Kohonen (SOM)

**GRUPO 10 · "Mapas Auto-Organizados de Kohonen (SOM): Redes que Ordenan Solas"**

**ESTA es la demo principal ahora** (`ejercicio_numerico_som.py`): resuelve un
**ejercicio numérico completo** del algoritmo de Kohonen donde **cada fórmula
se aplica con números visibles**, paso a paso y de forma interactiva.
La demo anterior (visual/noticiera) sigue disponible en
`demostracion_som.py`.

---

## ▶ Cómo ejecutarla

```bash
cd ~/Documentos/DEMOSTRACION\ -\ KOHOSEN
python3 ejercicio_numerico_som.py
```

Requisitos (ya instalados): `numpy` y `matplotlib`.

Al abrir aparece el **menú** con 3 secciones:

| Botón | Qué muestra |
|---|---|
| **1 · EJERCICIO PASO A PASO** | el ejercicio completo, fórmula por fórmula, con botones |
| **2 · CONVERGENCIA (24 épocas)** | el mismo ejercicio entrenando en automático |
| **3 · CHULETA DE FÓRMULAS** | las 5 fórmulas del SOM en una lámina |

Cierra la ventana de cada sección para volver al menú; cierra el menú para salir.

---

## 📐 El ejercicio (todo calculado con números)

**Planteamiento**: 4 datos en 2 dimensiones, mapa de **2×2 neuronas**,
η₀ = 0.6, σ₀ = 1.5, T = 16 pasos (4 épocas × 4 datos):

| dato | vector | grupo real (solo color) |
|---|---|---|
| X₁ | (0.2, 0.8) | A |
| X₂ | (0.3, 0.9) | A |
| X₃ | (0.8, 0.2) | B |
| X₄ | (0.9, 0.3) | B |

Pesos iniciales: W₁₁=(0.5, 0.5), W₁₂=(0.1, 0.2), W₂₁=(0.9, 0.8), W₂₂=(0.4, 0.1).

**Por cada dato se avanzan 5 pasos** (pulsando ▸ se ve la fórmula general
Y los números sustituidos):

1. **ENTRADA** — se presenta x = (0.200, 0.800) (normalizado a [0,1]²).
2. **COMPETENCIA** — `d(x, W) = √((x₁−W₁)² + (x₂−W₂)²)` calculada para las
   4 neuronas; la menor es la **BMU** (p. ej. d(x,W₁₁)=√(0.090+0.090)=**0.424**).
3. **COOPERACIÓN** — `h = exp(−d(r, r_BMU)²/(2σ²))`: vecindario gaussiano
   (p. ej. h₁₂ = exp(−1.000/4.500) = **0.801**).
4. **ADAPTACIÓN** — regla de Kohonen `W(nuevo) = W + η·h·(x−W)`, neurona por
   neurona (p. ej. ΔW₁₁ = 0.600·(−0.300,+0.300) → W₁₁ = (0.320, 0.680)).
5. **CONVERGENCIA** — decaída `η(t)=η₀·(1−t/T)` y `σ(t)=σ₀·(1−t/T)`.

Al final: **X₁,X₂ → W₁₁ (grupo A)** y **X₃,X₄ → W₂₂ (grupo B)**, pureza 100%,
**sin haber usado las etiquetas**.

---

## 🖱 Controles interactivos (ventana 1)

| Control | Acción |
|---|---|
| **▸ SIGUIENTE PASO** (o ESPACIO / tecla →) | avanza UN paso: muestra fórmula + números |
| **⏩ AUTO / ⏸ PAUSA** | reproduce todos los pasos automáticamente |
| **⟲ REINICIAR** (o tecla R) | empieza de cero |
| **✕ TERMINAR** (o cerrar ventana) | vuelve al menú |

Cada ventana tiene 3 vistas sincronizadas:

- **MAPA 2×2** — tarjetas con el vector de pesos de cada neurona; la BMU
  se marca con ★ dorada y su vecindario se sombrea con la intensidad de h;
  los puntitos de color muestran qué datos guardó cada neurona.
- **ESPACIO DE DATOS** — los datos (círculos de color) y los pesos
  (cuadrados dorados) **migrando** con flechas en cada adaptación.
- **PANEL DE CÁLCULOS** — la fórmula general + los números del paso actual.
- **Barras η/σ** — la decaída de los parámetros en vivo.

Además **cada paso queda impreso en la terminal** (útil para copiar los
números al informe o memorias de cálculo).

---

## 🎥 Consejos para presentar el ejercicio

1. Abre la **chuleta** primero (30 s) para fijar las 5 fórmulas.
2. En el ejercicio, avanza con ▸ y **pregunta al público** antes de cada
   paso: "¿quién ganará la competencia?" / "¿cuánto valdrá h?".
3. Señala en el ESPACIO DE DATOS las **flechas doradas**: los pesos
   físicamente se mueven hacia los datos — eso ES aprender.
4. Cierra con la **CONVERGENCIA**: en 24 épocas se ve el mapa ordenarse solo.
5. Remata: "100% de pureza **sin etiquetas** — eso es auto-organización".

## 🩺 Solución de problemas

| Síntoma | Solución |
|---|---|
| No se abre ventana | la demo cae a modo terminal y resuelve todo con números; para GUI: `sudo apt install python3-gi gir1.2-gtk-3.0` |
| Fuente rara en √ ₁ ² | usa la fuente DejaVu Sans (ya es la predeterminada) |
| Va muy rápido en AUTO | sube el `% 16` del bucle a `% 30` en `fase_ejercicio` |
