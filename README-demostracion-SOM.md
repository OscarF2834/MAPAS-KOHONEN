# 🗺️ Demostración interactiva · Mapas Auto-Organizados de Kohonen (SOM)

**"Redes que Ordenan Solas"** — GRUPO 10 · Octubre 26, 2026

Demostración visual en Python que acompaña la presentación. Todo el
algoritmo SOM (BMU, vecindario gaussiano, decaimiento de η y σ, matriz-U)
está implementado **desde cero** en un solo archivo, usando únicamente
`numpy` y `matplotlib` (ya instalados en tu paquete con pip).

## ▶ Cómo ejecutarla

```bash
cd ~/Documentos/DEMOSTRACION\ -\ KOHOSEN
python3 demostracion_som.py
```

Cada una de las **9 fases** abre una ventana animada. Ídeas siguiendo las
indicaciones de la terminal:

1. Cierra la ventana de la fase actual.
2. Pulsa **ENTER** en la terminal.
3. Pasa a la siguiente fase.

## 🎞️ Contenido (una fase por diapositiva del PDF)

| Fase | Tema de la diapositiva | Lo que muestra |
|------|------------------------|----------------|
| 1 | ¿Qué es un SOM? | El mapa "se ordena solo" con datos de colores: auto-organiza, reduce dimensiones, agrupa por similitud |
| 2 | Cuatro movimientos | Paso a paso animado: ENTRADA → COMPETENCIA → COOPERACIÓN → CONVERGENCIA |
| 3 | Muchas dimensiones | "Cerca en el mapa ≈ cerca en los datos" (correlación incluida) |
| 4 | Caso real 01 · MEDIOS | 132 "noticias" de 6 temas que se clasifican solas (pureza ~100%) |
| 5 | Caso real 02 · SEGURIDAD | Anomalías que destacan antes de tener nombre |
| 6 | ¿Por qué elegir un SOM? | Sin etiquetas · visual · flexible · robusto (4 mini-experimentos) |
| 7 | El mapa exige criterio | Comparación 5×5 / 10×10 / 22×22 y las 4 tensiones |
| 8 | El SOM del futuro | Mapa "vivo" que se adapta a un flujo que cambia (drift) |
| 9 | Cierre | ORDENA · REVELA · INVITA — equipo KOHOSEN |

## 🔧 Personaliza

Todo está al inicio del archivo `demostracion_som.py`:

- `PAUSA_MS` / `REFRESCO_CADA` → velocidad de las animaciones.
- Los datasets, tamaños de mapa (`SOM(filas, columnas, dim)`) y épocas
  de cada fase se cambian en un solo lugar.
