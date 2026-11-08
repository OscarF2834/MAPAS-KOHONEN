#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D E M O S T R A C I Ó N   D E   M A P A S   A U T O - O R G A N I Z A D O S
                     de Kohonen — SOM (Kohonen, 1982)

Demostración visual para la presentación
"Mapas Auto-Organizados de Kohonen (SOM): Redes que Ordenan Solas"
(GRUPO 10 · Octubre 26, 2026)

La red SOM está implementada DESDE CERO: solo usa numpy y matplotlib.
La BMU, el vecindario gaussiano, la decaída de la tasa de aprendizaje
y del radio están escritas aquí para que cada paso del algoritmo sea
visible y comprensible.

PASOS DE LA DEMO (cada fase abre una ventana; ciérrala y pulsa ENTER
en la terminal para pasar a la siguiente):

  [1] ¿Qué es un SOM?              → animación: el mapa "se ordena solo" con datos de colores
  [2] De un dato al mapa:          → los 4 movimientos paso a paso:
      cuatro movimientos            ENTRADA · COMPETENCIA · COOPERACIÓN · CONVERGENCIA
  [3] Muchas dimensiones,          → reducción de dimensión:
      un mapa legible               "cerca en el mapa ≈ cerca en los datos"
  [4] Caso real 01 · MEDIOS        → "noticias" que se agrupan solas por afinidad temática
  [5] Caso real 02 · SEGURIDAD     → anomalías que destacan antes de tener nombre
  [6] ¿Por qué elegir un SOM?      → sin etiquetas · visual · flexible · robusto
  [7] El mapa también exige        → tamaño del mapa, interpretación,
      criterio                        escalabilidad, heterogeneidad
  [8] El SOM del futuro            → mapa "vivo" que se adapta a un flujo en tiempo real
  [9] Cierre                       → ORDENA · REVELA · INVITA

Ejecutar:    python3 demostracion_som.py
"""

import math
import numpy as np
import matplotlib
matplotlib.use("GTK3Agg")          # backend gráfico disponible en este equipo
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"

# ──────────────────────────────────────────────────────────────────────────────
#  PALETA Y ESTILO — la demo usa la estética de la presentación
# ──────────────────────────────────────────────────────────────────────────────
AZUL     = "#16233f"   # fondo
AZUL_2   = "#1f3357"   # tarjeta
CREMA    = "#f5f1e8"   # texto claro
DORADO   = "#e8b64a"   # acento
TURQUESA = "#2ec4b6"
CORAL    = "#ef476f"
LILA     = "#b388eb"
VERDE    = "#7dd463"
GRIS     = "#8a93a5"
ROJO     = "#ff6b6b"

PALETA_TOP = [TURQUESA, LILA, CORAL, DORADO, VERDE,
              "#5bc0de", "#ff9f68", "#8f7de8"]

# paleta "matriz-U": oscuro (zona homogénea) → claro/peligro (frontera)
from matplotlib.colors import LinearSegmentedColormap
UM_CMAP = LinearSegmentedColormap.from_list(
    "um", ["#101a30", "#1e4d5e", "#2ea98f", "#d8b64a", "#f2f1e8"])

PAUSA_MS = 0.001            # refresco mínimo del bucle gráfico
REFRESCO_CADA = 14          # cuántos pasos de entrenamiento por redraw


# ══════════════════════════════════════════════════════════════════════════════
#  N Ú C L E O  —  LA RED SOM ESCRITA DESDE CERO
# ══════════════════════════════════════════════════════════════════════════════
class SOM:
    """
    Mapa Auto-Organizado de Kohonen, con los pasos VISIBLES.

    La red es una cuadrícula filas×columnas de neuronas; cada neurona
    guarda un vector de pesos de la misma dimensión que los datos.
    """

    def __init__(self, filas=9, columnas=9, dim=3, semilla=7):
        rng = np.random.default_rng(semilla)
        self.filas, self.columnas, self.dim = filas, columnas, dim
        self.W = rng.random((filas, columnas, dim))          # pesos
        I, J = np.mgrid[0:filas, 0:columnas]
        self.pos = np.stack([I, J], axis=-1)                 # (f,c,2)

    # ── PASO 1 · ENTRADA ────────────────────────────────────────────────────
    @staticmethod
    def normalizar(X):
        """Escala cada rasgo a [0,1]. El SOM exige comparar 'peras con peras'."""
        mn, mx = X.min(0), X.max(0)
        r = np.where(mx - mn == 0, 1.0, mx - mn)
        return (X - mn) / r

    # ── PASO 2 + PASO 3: un paso de aprendizaje devuelto VISIBLE ───────────
    def paso(self, x, eta, sigma):
        """
        Un paso de entrenamiento. Devuelve (bmu, d, G) para que la demo
        pueda dibujar la competición y la cooperación en vivo.
           PASO 2 · COMPETENCIA : encontrar la BMU (mejor neurona)
           PASO 3 · COOPERACIÓN : mover a la BMU **y** a su vecindario
        """
        d = np.linalg.norm(self.W - x, axis=-1)              # distancia a cada neurona
        bmu = np.unravel_index(np.argmin(d), d.shape)        # → BMU
        dist2d = np.linalg.norm(self.pos - np.array(bmu), axis=-1)
        G = np.exp(-(dist2d ** 2) / (2 * sigma ** 2))        # función topológica
        self.W += eta * G[..., None] * (x - self.W)          # regla de Kohonen
        return bmu, d, G

    # stepper de conveniencia para entrenamientos rápidos
    def entrenar(self, X, epocas=40, t0=.5, r0=None, rng=None):
        n = len(X)
        if r0 is None:
            r0 = max(self.filas, self.columnas) / 2
        if rng is None:
            rng = np.random.default_rng(1)
        total = epocas * n
        k = 0
        for _ in range(epocas):
            for xi in rng.permutation(n):
                lit = 1 - k / total
                sigma = max(.65, r0 * lit)                   # PASO 4: radio decrece
                eta = max(.02, t0 * lit ** 1.6)              # PASO 4: tasa decrece
                self.paso(X[xi], eta, sigma)
                k += 1

    def bmu_de(self, x):
        """Competencia (solo lectura) para clasificar un dato nuevo."""
        d = np.linalg.norm(self.W - x, axis=-1)
        i = np.unravel_index(np.argmin(d), d.shape)
        return np.array(i), d[i]

    def matriz_u(self):
        """Distancia media a las vecinas: fronteras claras, grupos oscuros."""
        f, c = self.filas, self.columnas
        um = np.zeros((f, c))
        for i in range(f):
            for j in range(c):
                ds = []
                for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    a, b = i + di, j + dj
                    if 0 <= a < f and 0 <= b < c:
                        ds.append(np.linalg.norm(self.W[i, j] - self.W[a, b]))
                if ds:
                    um[i, j] = np.mean(ds)
        return um / max(um.max(), 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
#  utilidades de presentación
# ══════════════════════════════════════════════════════════════════════════════
def aviso(t):
    print("\n  \033[96m▸\033[0m " + t)

def pausa_enter(sugerencia="cierra la ventana y descubro la siguiente fase"):
    print("    \033[90m► " + sugerencia +
          ", luego pulsa ENTER aquí…\033[0m ", end="")
    input("")
    print()

def figura_presentation(w=15.5, h=9.0):
    """Figura con el marco y el fondo de la presentación."""
    fig = plt.figure(figsize=(w, h), facecolor=AZUL)
    ax0 = fig.add_axes([0, 0, 1, 1], zorder=-20)
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1); ax0.axis("off")
    ax0.add_patch(FancyBboxPatch(
        (0.010, 0.010), 0.980, 0.980,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        fc=AZUL_2, ec=DORADO, lw=2.4, alpha=.30, zorder=-20))
    ax0.text(.985, .022, "SOM · Kohonen 1982", fontsize=8.5, color=GRIS,
             ha="right", va="bottom", zorder=-20)
    return fig, ax0

def titulo_slide(ax0, es, sub=None):
    ax0.text(.046, .955, es, fontsize=27, color=CREMA,
             fontweight="bold", va="top")
    if sub:
        ax0.text(.046, .905, sub, fontsize=13, color=DORADO, va="top")

def chip(ax, x, y, tx, color, fs=11.5, w=.20, h=.078, lw=2.0, border=False):
    """Cajita 'chip' con borde redondeado tipo presentación."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.022",
        fc=(AZUL if border else "none"), ec=color, lw=lw))
    ax.text(x + w / 2, y + h / 2, tx, fontsize=fs, color=CREMA,
            ha="center", va="center", fontweight="bold")

def tarjeta(ax, x, y, w, h, color):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.022",
        fc=AZUL, ec=color, lw=2.2))

def panel_oscuro(ax, titulo=None, lim=None):
    ax.set_facecolor("#141f38")
    for NombreSpine in ax.spines.values():
        NombreSpine.set_color("#2b3d63")
    ax.tick_params(colors=GRIS, labelsize=9)
    if titulo:
        ax.set_title(titulo, color=CREMA, fontsize=11, pad=10,
                     fontweight="bold")
    if lim is not None:
        ax.set_xlim(lim[0]); ax.set_ylim(lim[1])

class Ventana:
    """
    Pequeño envoltorio: sabe si el usuario cerró la ventana y ofrece un
    'refrescar' uno-a-uno para las animaciones con plt.pause.
    """
    def __init__(self, fig):
        self.fig = fig
        self.cerrada = False
        fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "cerrada", True))
        plt.show(block=False)

    def refrescar(self, pausa=PAUSA_MS):
        if self.cerrada:
            return False
        try:
            self.fig.canvas.draw_idle()
            plt.pause(pausa)
            return True
        except Exception:
            self.cerrada = True
            return False

def cerrar(fen):
    try:
        plt.close(fen.fig)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 1 — ¿QUÉ ES UN MAPA AUTO-ORGANIZADO?
# ══════════════════════════════════════════════════════════════════════════════
def datos_colores(n=380, semilla=9):
    """
    Datos 'de juguete': colores del cubo RGB. Intuitivos — si el SOM
    funciona, el mapa termina ordenado por tonos, con los colores puros
    (rojo, verde, azul, amarillo…) en su lugar del mapa.
    """
    rng = np.random.default_rng(semilla)
    X = list(rng.random((n, 3)))
    for base, k in (((1, .10, .10), 45), ((.05, .85, .25), 45),
                    ((.12, .18, 1.0), 45), ((1.0, 1.0, .05), 36),
                    ((1.0, .45, .02), 36), ((1.0, .8, .8), 25),
                    ((.75, .75, .75), 25), ((0.0, 0.0, 0.0), 25)):
        X.append(np.clip(np.array(base) + rng.normal(0, .045, (k, 3)), 0, 1))
    return np.vstack(X)


def celdas_mapa(ax, som):
    """Dibuja la cuadrícula del SOM coloreando cada neurona con su vector."""
    f, c = som.filas, som.columnas
    rets = []
    for i in range(f):
        for j in range(c):
            r = Rectangle((j, i), .92, .92, fc=np.clip(som.W[i, j], 0, 1),
                          ec="none")
            rets.append(r)
            ax.add_patch(r)
    ax.set_xlim(-.1, c + .05); ax.set_ylim(-.1, f + .05)
    ax.axis("off"); ax.invert_yaxis()
    ax.set_aspect("equal")
    return rets


def fase1():
    aviso("FASE 1 · ¿Qué es un Mapa Auto-Organizado?")
    fig, ax0 = figura_presentation()
    titulo_slide(ax0, "¿Qué es un Mapa Auto-Organizado?",
                 "Red neuronal no supervisada que aprende SIN etiquetas: "
                 "lo parecido termina cerca; lo distinto, lejos.")
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1],
                  left=.045, right=.955, top=.78, bottom=.10, wspace=.20)
    axmap = fig.add_subplot(gs[0])
    axtxt = fig.add_subplot(gs[1])

    for ax in (axmap, axtxt):
        ax.set_facecolor(AZUL)
    panel_oscuro(axmap, "DATOS DE ENTRADA (3D)  →  CUADRÍCULA 12×12")
    axtxt.axis("off")

    # contacto visual: 3 ventajas, tal como en la diapositiva
    axtxt.set_xlim(0, 1); axtxt.set_ylim(0, 1)
    axtxt.text(.5, .96, "EN 3 IDEAS", color=DORADO, fontsize=14,
               fontweight="bold", ha="center", va="top")
    ideas = [
        ("AUTO-ORGANIZA", "ningún humano le dicta cómo agrupar.\n"
                          "La estructura EMERGE de los propios datos.", TURQUESA),
        ("REDUCE DIMENSIONES", "vectores de 3 (o 300) dimensiones\n"
                               "quedan proyectados en una malla 12×12.", LILA),
        ("AGRUPA POR SIMILITUD", "observaciones parecidas caen\n"
                                 "cerca de sí mismas en el mapa.", CORAL),
    ]
    y = .90
    for t, d, c in ideas:
        axtxt.text(.05, y, "◆", color=c, fontsize=17, va="top")
        axtxt.text(.12, y, t, color=CREMA, fontsize=13, fontweight="bold", va="top")
        axtxt.text(.12, y - .055, d, color=GRIS, fontsize=10.5, va="top")
        y -= .20

    # barra de progreso
    axtxt.text(.05, .335, "estado del entrenamiento", color=GRIS, fontsize=10,
               fontweight="bold")
    jan = .02
    axtxt.add_patch(FancyBboxPatch((jan, .295), .96, .05,
                   boxstyle="round,pad=0.006,rounding_size=0.02",
                   fc="#0e172c", ec="#2b3d63", lw=1.2))
    progreso = FancyBboxPatch((jan + .006, .302), .0005, .036,
                              boxstyle="round,pad=0,rounding_size=0.015",
                              fc=DORADO, ec="none")
    axtxt.add_patch(progreso)

    # ----Panel 1 los datos "crudos" en miniatura antes del mapa ----------------
    axmini = fig.add_axes([.565, .075, .295, .170])   # miniatura bajo el panel
    axmini.set_facecolor("#0e172c")
    axmini.set_title("así LLEGAN los datos (sin estructura)", color=GRIS,
                     fontsize=9, pad=5)
    datos = datos_colores()
    rngd = np.random.default_rng(3)
    desorden = rngd.permutation(len(datos))[:260]
    axmini.scatter(
        rngd.random(260), rngd.random(260),
        c=datos[desorden], s=34, lw=0, alpha=.9)
    axmini.set_xticks([]); axmini.set_yticks([])
    axmini.invert_yaxis()

    # ---- SOM -------------------------------------------------------------------
    som = SOM(12, 12, 3, semilla=5)
    Xn = som.normalizar(datos)
    cuer = celdas_mapa(axmap, som)

    tar = Ventana(fig)
    epocas = 34
    total = epocas * len(Xn)
    r0, t0 = 7.0, .55
    paso = 0

    for ep in range(epocas):
        for xi in np.random.default_rng(11 + ep).permutation(len(Xn)):
            lit = 1 - paso / total
            sigma = max(.75, r0 * lit)
            eta = max(.03, t0 * lit ** 1.6)
            som.paso(Xn[xi], eta, sigma)
            paso += 1
            if paso % REFRESCO_CADA == 0 or paso == total:
                for idx, r in enumerate(cuer):
                    i, j = divmod(idx, som.columnas)
                    r.set_facecolor(np.clip(som.W[i, j], 0, 1))
                progreso.set_width(.945 * paso / total)
                lit01 = 1 - paso / total
                axmap.set_xlabel(
                    f"época {ep+1}/{epocas}   "
                    f"PASO 4 ▸  η (tasa) = {max(.03, t0*lit01**1.6):.3f}  ·  "
                    f"σ (radio) = {max(.75, r0*lit01):.2f}  (cada vez más pequeños)",
                    color=CREMA, fontsize=10)
                if not tar.refrescar():
                    break
        if tar.cerrada:
            break

    # una vez ordenado, muestra los datos en su celda
    if not tar.cerrada:
        rngb = np.random.default_rng(4)
        for x in Xn[:300]:
            bmu, _ = som.bmu_de(x)
            axmap.scatter(bmu[1] + .46 + rngb.normal(0, .13),
                          bmu[0] + .46 + rngb.normal(0, .13),
                          c=[np.clip(x, 0, 1)], s=12, lw=0, alpha=.55, zorder=4)
        axmap.set_title("DESPUÉS: cada punto encontró su lugar")
        progreso.set_width(.945)
        while not tar.cerrada:
            tar.refrescar(0.02)
        cerrar(tar)

    print("    Fin de la fase 1.")


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 2 — DE UN DATO AL MAPA: CUATRO MOVIMIENTOS
# ══════════════════════════════════════════════════════════════════════════════
def fase2():
    aviso("FASE 2 · De un dato al mapa: cuatro movimientos")
    fig, ax0 = figura_presentation()
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.15, 1],
                  height_ratios=[3.6, 1.15], left=.045, right=.95,
                  top=.80, bottom=.075, wspace=.16, hspace=.32)
    axmap = fig.add_subplot(gs[0, 0])
    axrt  = fig.add_subplot(gs[:, 1])
    axsab = fig.add_subplot(gs[1, 0])
    for ax in (axmap, axrt, axsab):
        ax.set_facecolor(AZUL)

    panel_oscuro(axmap, "cuando llega UN dato… ¿qué hace la red?")
    axmap.axis("off"); axrt.axis("off")
    axmap.invert_yaxis()

    som = SOM(7, 7, 3, semilla=11)
    datos = datos_colores(n=260, semilla=21)
    Xn = som.normalizar(datos)
    # pre-entrenamiento silencioso para que «a mitad de convergencia» la
    # retracción se note con cada dato nuevo
    som.entrenar(Xn, epocas=120, t0=.5, r0=4.0, rng=np.random.default_rng(2))

    cua = celdas_mapa(axmap, som)

    # tarjetas las 4 etapas
    axrt.set_xlim(0, 1); axrt.set_ylim(0, 1)
    tarjetas_spec = [
        ("1 · ENTRADA", "El dato se NORMALIZA: poner a todos los rasgos\n"
                        "en la misma escala (aquí son 3 colores).", TURQUESA),
        ("2 · COMPETENCIA", "Las neuronas «compiten»: gana la que más\n"
                            "se parece al dato → se llama BMU.", CORAL),
        ("3 · COOPERACIÓN", "La BMU NO va sola: su vecindario gaussiano\n"
                            "se mueve con ella. Se aprende en equipo.", LILA),
        ("4 · CONVERGENCIA", "Con el tiempo, la tasa de aprendizaje η y el\n"
                             "radio σ DECRECEN → el mapa se estabiliza.", DORADO),
    ]
    tar_rects = []
    y = .775
    for t, d, c in tarjetas_spec:
        tarjeta(axrt, .03, y, .94, .205, c)
        axrt.text(.085, y + .17, t, color=c, fontsize=13.5, va="center",
                  fontweight="bold")
        axrt.text(.085, y + .075, d, color=CREMA, fontsize=10.5, va="center")
        tar_rects.append(axrt.patches[-1])
        y -= .215

    # --- animación ----------------------------------------------------------
    punto = Circle((0, 0), .12, fc="white", ec=DORADO, lw=3, alpha=0,
                   zorder=6)                                    # dato entrante
    vecindario = Circle((0, 0), 0.0, fc="none", ec=LILA, lw=3,
                        ls="--", alpha=0, zorder=5)             # radio σ
    estrella = axmap.plot([], [], marker="*", ms=26, color=DORADO,
                          mec="#fff", mew=1.2, ls="", alpha=0, zorder=7)[0]
    axmap.add_patch(punto); axmap.add_patch(vecindario)

    # barras de η y σ en el panel inferior (PASO 4)
    axsab.set_xlim(0, 1); axsab.set_ylim(0, 1)
    axsab.axis("off")
    axsab.text(.50, .93, "PASO 4 · CONVERGENCIA: η y σ decaen con el tiempo",
               color=CREMA, fontsize=11.5, fontweight="bold", ha="center")
    axsab.add_patch(FancyBboxPatch((.05, .58), .90, .11,
                   boxstyle="round,pad=0.005,rounding_size=0.015",
                   fc="#0e172c", ec="#2b3d63", lw=1))
    axsab.add_patch(FancyBboxPatch((.05, .20), .90, .11,
                   boxstyle="round,pad=0.005,rounding_size=0.015",
                   fc="#0e172c", ec="#2b3d63", lw=1))
    eta_bar2 = FancyBboxPatch((.055, .587), .001, .096,
                              boxstyle="round,pad=0,rounding_size=0.012",
                              fc=CORAL, ec="none")
    sig_bar2 = FancyBboxPatch((.055, .227), .001, .096,
                              boxstyle="round,pad=0,rounding_size=0.012",
                              fc=LILA, ec="none")
    axsab.add_patch(eta_bar2); axsab.add_patch(sig_bar2)
    axsab.text(.05, .82, "η tasa de aprendizaje", color=CORAL, fontsize=10,
               fontweight="bold")
    axsab.text(.05, .44, "σ radio de vecindad", color=LILA, fontsize=10,
               fontweight="bold")

    tar = Ventana(fig)
    # script: cada dato pasa por los 4 movimientos, uno a uno
    N_DATOS = 26
    indices = np.random.default_rng(8).choice(len(Xn), N_DATOS, replace=False)
    t_s, t_e = 1.6, 0.8          # σ y η iniciales en la fase visible

    acabado = False
    for kk, idx in enumerate(indices):
        if tar.cerrada or acabado:
            break
        x = Xn[idx]
        # momento global de la vida de la red
        fracc = (kk + 1) / N_DATOS
        eta  = t_e * (1 - 0.88 * fracc)
        sigma = t_s * (1 - 0.75 * fracc)

        # ★ 1 · ENTRADA: el dato 'aterriza'
        punto.set_facecolor(np.clip(x, 0, 1)); punto.set_alpha(1)
        punto.center = (3.0, -0.68)
        for rI in tar_rects: rI.set_linewidth(2.2)
        tar_rects[0].set_linewidth(5.0)
        for _ in range(10):
            if not tar.refrescar(): acabado = True; break

        if acabado: break
        # ★ 2 · COMPETENCIA: se elige la BMU
        bmu, d, G = som.paso(x, eta, sigma)
        p_c = (bmu[1] + .46, bmu[0] + .46)
        punto.center = p_c
        estrella.set_data([bmu[1] + .46], [bmu[0] + .46]); estrella.set_alpha(1)
        tar_rects[0].set_linewidth(2.2)
        tar_rects[1].set_linewidth(5.0)
        for _ in range(12):
            if not tar.refrescar(): acabado = True; break

        if acabado: break
        # ★ 3 · COOPERACIÓN: se mueve el vecindario
        vecindario.center = p_c
        vecindario.set_radius(sigma)
        vecindario.set_alpha(.85)
        tar_rects[1].set_linewidth(2.2)
        tar_rects[2].set_linewidth(5.0)
        for ii in range(8):
            vecindario.set_alpha(.85 - ii * .06)
            if not tar.refrescar(): acabado = True; break

        if acabado: break
        # ★ 4 · CONVERGENCIA: barras de η y σ
        tar_rects[2].set_linewidth(2.2)
        tar_rects[3].set_linewidth(5.0)
        eta_bar2.set_width(.885 * eta / t_e * fracc + .001)
        sig_bar2.set_width(.885 * sigma / t_s * fracc + .001)
        point = None
        for idx2, r in enumerate(cua):
            i, j = divmod(idx2, som.columnas)
            r.set_facecolor(np.clip(som.W[i, j], 0, 1))
        axsab.text(.955, .635, f"η={eta:.3f}", color=CORAL, fontsize=9,
                   ha="right",
                   fontweight="bold")
        axsab.text(.955, .262, f"σ={sigma:.2f}", color=LILA, fontsize=9,
                   ha="right", fontweight="bold")
        for _ in range(11):
            if not tar.refrescar(): acabado = True; break

        punto.set_alpha(0); estrella.set_alpha(0)
        vecindario.set_alpha(0)

    aviso("fin del entrenamiento: mapa estabilizado.")
    while not tar.cerrada:
        tar.refrescar(0.02)
    cerrar(tar)
    print("    Fin de la fase 2.")


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 3 — MUCHAS DIMENSIONES, UN MAPA LEGIBLE
# ══════════════════════════════════════════════════════════════════════════════
def fase3():
    aviso("FASE 3 · Muchas dimensiones, un mapa legible")
    fig, ax0 = figura_presentation()
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1],
                  height_ratios=[1.05, 1], left=.06, right=.94,
                  top=.80, bottom=.10, wspace=.25, hspace=.42)
    axorig = fig.add_subplot(gs[0, 0])
    axmap  = fig.add_subplot(gs[0, 1])
    axcorr = fig.add_subplot(gs[1, :])

    for ax in (axorig, axmap, axcorr):
        ax.set_facecolor(AZUL)

    titulo_slide(ax0, "Muchas dimensiones. Un mapa legible.",
                 "El SOM proyecta cada vector sobre una cuadrícula 2D "
                 "conservando, en lo posible, quién se parece a quién.")

    # datos con estructura en 3D: 4 grupos bien separados + tejido común
    rng = np.random.default_rng(77)
    centros = np.array([[ .8, .2, .2], [.2, .8, .2],
                        [.2, .2, .8], [.8, .8, .8]])
    etiquetas = [0, 1, 2, 3]
    X = []
    y = []
    for kk, c in enumerate(centros):
        X.append(np.clip(rng.normal(c, .085, (70, 3)), 0, 1))
        y += [kk] * 70
    X.append(np.clip(rng.random((60, 3)) * .25 + .38, 0, 1))
    y += [4] * 60
    X = np.vstack(X); y = np.array(y)
    Xn = SOM.normalizar(X)

    som = SOM(11, 11, 3, semilla=20)
    som.entrenar(Xn, epocas=70, t0=.5, r0=5.0, rng=np.random.default_rng(6))

    # (A) el espacio original, proyectado en 2D (debe verse como nube caótica)
    axorig.scatter(X[:, 0], X[:, 1], c=X, s=22, lw=0, alpha=.75)
    panel_oscuro(axorig, "ESPACIO ORIGINAL: 3 dimensiones (nube caótica)")
    axorig.set_xlabel("dimensión 1", color=GRIS, fontsize=9.5)
    axorig.set_ylabel("dimensión 2", color=GRIS, fontsize=9.5)

    # (B) el territorio: cada neurona con el color MEDIO de sus hits
    bmus = [som.bmu_de(x)[0] for x in Xn]
    conteo = np.zeros((11, 11)); ac = np.zeros((11, 11, 3))
    for b, x in zip(bmus, Xn):
        conteo[tuple(b)] += 1
        ac[tuple(b)] += x
    visu = np.full((11, 11, 3), 0.24)
    for i in range(11):
        for j in range(11):
            if conteo[i, j]:
                visu[i, j] = np.clip(ac[i, j] / conteo[i, j], 0, 1)
    axmap.imshow(visu, extent=[-.5, 10.5, 10.5, -.5])
    panel_oscuro(axmap, "EL MAPA 2D: cada color = grupo emergente")
    axmap.set_xlabel("cuadrícula 11×11", color=GRIS, fontsize=9.5)

    # (C) la PROMESA: cerca en el mapa ≈ cerca en los datos
    idx = rng.choice(len(Xn), 130, replace=False)
    ds_dat, ds_map = [], []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            ds_dat.append(np.linalg.norm(Xn[i] - Xn[j]))
            ds_map.append(np.linalg.norm(bmus[i] - bmus[j]))
    ds_dat, ds_map = np.array(ds_dat), np.array(ds_map)
    r = np.corrcoef(ds_dat, ds_map)[0, 1]
    axcorr.scatter(ds_dat, ds_map, s=7, c=TURQUESA, alpha=.45, lw=0)
    panel_oscuro(axcorr,
                 "LA PROMESA:  «cerca en el mapa  ≈  cerca en los datos»")
    axcorr.set_xlabel("distancia en los DATOS (3D, normalizada)", color=GRIS)
    axcorr.set_ylabel("distancia en el MAPA (casillas)", color=GRIS)

    m, b = np.polyfit(ds_dat, ds_map, 1)
    xs = np.linspace(0, ds_dat.max(), 50)
    axcorr.plot(xs, m * xs + b, color=DORADO, lw=2.4,
                label=f"correlación  r = {r:.2f}")
    axcorr.legend(facecolor=AZUL_2, edgecolor="#2b3d63", labelcolor=CREMA,
                  fontsize=10)

    tarjeta(ax0, .18, .015, .64, .052, CORAL)
    ax0.text(.50, .041, "multitud de rasgos (aquí 3, en la vida real 300)  →  "
                        "2 dimensiones EXPLORABLES",
             color=CREMA, fontsize=12, ha="center", va="center",
             fontweight="bold")

    tar = Ventana(fig)
    while not tar.cerrada:
        tar.refrescar(0.03)
    cerrar(tar)
    print(f"    (correlación datos/mapa r = {r:.2f})")
    print("    Fin de la fase 3.")


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 4 — CASO REAL 01 · MEDIOS: NOTICIAS QUE SE ORDENAN SOLAS
# ══════════════════════════════════════════════════════════════════════════════
def fase4():
    aviso("FASE 4 · Caso real 01 — Noticias que se ordenan solas")
    fig, ax0 = figura_presentation()
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1],
                  left=.045, right=.957, top=.82, bottom=.08, wspace=.20)
    axmap = fig.add_subplot(gs[0])
    axtxt = fig.add_subplot(gs[1])
    for ax in (axmap, axtxt):
        ax.set_facecolor(AZUL)
    axtxt.axis("off")

    # 6 categorías sin etiquetas visibles para el SOM: 36 palabras clave
    TEMAS = {
        "deportes":  ["futbol", "partido", "final", "jugador", "campeon", "equipo"],
        "tecnologia":["datos", "software", "red", "aplicacion", "robot", "memoria"],
        "politica":  ["parlamento", "voto", "diputado", "debate", "gobierno",
                      "ley"],
        "economia":  ["mercado", "acciones", "inflacion", "banco", "dolar",
                      "comercio"],
        "ciencia":   ["estrella", "galaxia", "molecula", "experimento",
                      "laboratorio", "planeta"],
        "espectaculos":["cine", "festival", "actor", "musica", "estreno",
                        "concierto"],
    }
    RELLENOS = ["ayer", "todavia", "esta", "del", "aunque", "porque", "sea",
                "habria", "frente", "sobre", "entre", "tambien"]
    vocab = sorted({w for tws in TEMAS.values() for w in tws} |
                   set(RELLENOS))
    widx = {w: k for k, w in enumerate(vocab)}
    k_temas = list(TEMAS)

    def vector_doc(doc):
        v = np.zeros(len(vocab))
        for w in doc:
            v[widx[w]] += 1
        return v / max(np.linalg.norm(v), 1e-9)

    rng = np.random.default_rng(31)
    docs, etiq = [], []
    for tema, palabras in TEMAS.items():
        for _ in range(22):
            n = rng.integers(3, 6)
            doc = list(rng.choice(palabras, n, replace=False))
            for _ in range(2):
                doc.append(rng.choice(RELLENOS))
            docs.append(doc); etiq.append(tema)
    Xn = np.array([vector_doc(d) for d in docs])

    som = SOM(10, 10, len(vocab), semilla=33)
    som.entrenar(Xn, epocas=60, t0=.45, r0=4.5,
                 rng=np.random.default_rng(12))

    # cada documento cae en su BMU; cada neurona toma el «color» dominante
    bmus = [tuple(som.bmu_de(x)[0]) for x in Xn]
    hit = {}
    for b, t in zip(bmus, etiq):
        hit.setdefault(b, {}).setdefault(t, 0)
        hit[b][t] += 1
    # pureza: ¿el grupo que EMERGIÓ en la celda coincide con la etiqueta?
    ok, tot = 0, 0
    for b, dist in hit.items():
        dom, nd = max(dist.items(), key=lambda kv: kv[1])
        tot += nd
        ok += dist.get(dom, 0)
    pureza = ok / tot

    um = som.matriz_u()
    NOMBRES = {"deportes": "deportes", "tecnologia": "tecnolog.",
               "politica": "política", "economia": "economía",
               "ciencia": "ciencia", "espectaculos": "espectácl."}
    axmap.imshow(um, cmap=UM_CMAP, extent=[-.5, 9.5, 9.5, -.5])
    panel_oscuro(axmap, "MATRIZ-U del mapa noticioso (fronteras en claro)")
    nombre_color = {k: PALETA_TOP[i] for i, k in enumerate(k_temas)}
    for b, dist in hit.items():
        if sum(dist.values()) >= 3:
            dom, nd = max(dist.items(), key=lambda kv: kv[1])
            axmap.text(b[1], b[0], NOMBRES[dom], color=CREMA,
                       fontsize=7.2, fontweight="bold",
                       ha="center", va="center",
                       bbox=dict(fc=nombre_color[dom], ec="none",
                                 boxstyle="round,pad=.20", alpha=.90, lw=0))

    titulo_slide(ax0, "CASO REAL 01 · MEDIOS",
                 "Un SOM agrupa miles de artículos por afinidad temática "
                 "SIN taxonomías manuales.")

    axtxt.set_xlim(0, 1); axtxt.set_ylim(0, 1)
    axtxt.text(.5, .95, "¿QUÉ PASÓ AQUÍ?", color=DORADO, fontsize=13.5,
               fontweight="bold", ha="center", va="top")
    axtxt.text(.03, .88,
               f"   Se generaron 132 pretendidos «titulares» de 6 temas.\n"
               f"   Cada uno se volvió un vector de {len(vocab)} rasgos\n"
               "   (frecuencia de cada palabra de vocabulario).",
               color=CREMA, fontsize=10.5, va="top")
    axtxt.text(.03, .72,
               "   El SOM los recibió SIN etiquetas y los organizó\n"
               "   solos: las zonas del mapa coinciden con las secciones\n"
               "   editoriales que creías que era difícil descubrir.",
               color=GRIS, fontsize=10.5, va="top")

    tarjeta(axtxt, .06, .50, .88, .10, VERDE)
    axtxt.text(.50, .55, f"PUREZA de los grupos emergentes: {pureza:.0%}",
               color=CREMA, fontsize=15, ha="center", va="center",
               fontweight="bold")
    axtxt.text(.06, .38, "El mapa ofrece NAVEGACIÓN intuitiva y revela\n"
                         "tendencias emergentes antes de que sean obvias.",
               color=GRIS, fontsize=10.5)
    ley = "leyenda de los temas (solo para TÍ — el SOM nunca las vio):"
    axtxt.text(.06, .22, ley, color=GRIS, fontsize=8.8, style="italic")
    xx, yy = .06, .17
    for k, kk in enumerate(k_temas):
        c = nombre_color[kk]
        axtxt.add_patch(Rectangle((xx, yy), .018, .032, fc=c, ec="none"))
        axtxt.text(xx + .026, yy + .016, kk.replace("_", " "), color=CREMA,
                   fontsize=9.5, va="center")
        xx += .315
        if k == 2:
            xx = .06; yy = .115

    tar = Ventana(fig)
    while not tar.cerrada:
        tar.refrescar(0.03)
    cerrar(tar)
    print(f"    pureza de los grupos emergentes ≈ {pureza:.0%}")
    print("    Fin de la fase 4.")


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 5 — CASO REAL 02 · SEGURIDAD: LO EXTRAÑO DESTACA ANTES DE TENER NOMBRE
# ══════════════════════════════════════════════════════════════════════════════
def fase5():
    aviso("FASE 5 · Caso real 02 — Anomalías: seguridad")
    fig, ax0 = figura_presentation()
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1],
                  left=.05, right=.955, top=.82, bottom=.15, wspace=.24)
    axmap = fig.add_subplot(gs[0])
    axqe = fig.add_subplot(gs[1])
    for ax in (axmap, axqe):
        ax.set_facecolor(AZUL)

    titulo_slide(ax0, "CASO REAL 02 · SEGURIDAD",
                 "El mapa aprende el patrón normal del tráfico. "
                 "Lo Extraño destaca antes de tener nombre.")

    # 5 rasgos sintéticos de 'tráfico'
    rng = np.random.default_rng(66)
    c_nor = np.array([[.85, .25, .2, .15, .35],
                      [.20, .50, .55, .45, .50],
                      [.55, .80, .30, .60, .75]])
    Xn = np.vstack([np.clip(rng.normal(c, .09, (150, 5)), 0, 1) for c in c_nor])
    Xani = np.zeros((30, 5))
    if True:
        # anomalías: picos de tasa, puertos raros, horas imposibles
        Xani[:, 0] = np.clip(rng.uniform(0, 1, 30), 0, 1)
        Xani[:, 1] = np.clip(rng.uniform(.92, 1, 30), 0, 1)
        Xani[:, 2] = np.clip(rng.uniform(.95, 1, 30), 0, 1)
        Xani[:, 3] = np.clip(rng.uniform(.85, 1, 30), 0, 1)
        Xani[:, 4] = np.clip(rng.uniform(0, 1, 30), 0, 1)

    som = SOM(9, 9, 5, semilla=40)
    som.entrenar(Xn, epocas=55, t0=.5, r0=4.2, rng=np.random.default_rng(9))

    qe_n = np.array([som.bmu_de(x)[1] for x in Xn])
    qe_a = np.array([som.bmu_de(x)[1] for x in Xani])
    umbral = qe_n.mean() + 2.6 * qe_n.std()

    um = som.matriz_u()
    axmap.imshow(um, cmap=UM_CMAP, extent=[-.5, 8.5, 8.5, -.5])
    panel_oscuro(axmap, "el territorio de lo «NORMAL»")
    rb = np.random.default_rng(15)
    for x, qe in zip(Xn[:240], qe_n[:240]):
        b, _ = som.bmu_de(x)
        axmap.scatter(b[1] + rb.normal(0, .16), b[0] + rb.normal(0, .16),
                      s=13, c=GRIS, alpha=.35, lw=0)
    n_marca = 0
    for x, qe in zip(Xani, qe_a):
        b, d = som.bmu_de(x)
        cls = CORAL if d > umbral else GRIS
        if d > umbral:
            n_marca += 1
        axmap.scatter(b[1], b[0], s=230, marker="X", c=cls,
                      edgecolors="#fff", lw=1.1, zorder=6, alpha=.95)
    axmap.set_xlabel("las X rojas son eventos que NO tienen casa en el mapa",
                     color=CREMA, fontsize=10)

    # radiografía del error de cuantización
    axqe.scatter(np.arange(len(qe_n)), qe_n, s=11, c=TURQUESA, alpha=0.55,
                 lw=0, label="eventos normales")
    axqe.scatter(np.arange(2000, 2000 + len(qe_a)), qe_a, s=85, marker="X",
                 c=CORAL, edgecolors="#fff", lw=.8,
                 label="eventos por CLASIFICAR")
    axqe.axhline(umbral, color=DORADO, lw=2.2, ls="--",
                 label="umbral de alarma (media + 2.6 σ)")
    panel_oscuro(axqe, "señal temprana: distancia a la BMU "
                       "(error de cuantización)")
    axqe.set_xlabel("evento (los últimos son los «raros»)", color=GRIS)
    axqe.set_ylabel("distancia al mapa", color=GRIS, fontsize=10)
    axqe.legend(facecolor=AZUL_2, edgecolor="#2b3d63", labelcolor=CREMA,
                fontsize=9, loc="center")

    axqe.text(.02, .98, f"{n_marca} de {len(qe_a)} eventos anómalos "
                        f"quedaron por ENCIMA del umbral",
              transform=axqe.transAxes, color=CREMA, fontsize=10.5,
              ha="left", va="top", fontweight="bold")

    # chip conclusión
    tarjeta(ax0, .16, .008, .68, .052, CORAL)
    ax0.text(.50, .034, "una señal temprana INCLUSO SIN ejemplos etiquetados "
                        "de ataques",
             color=CREMA, fontsize=12, ha="center", va="center",
             fontweight="bold")

    tar = Ventana(fig)
    while not tar.cerrada:
        tar.refrescar(0.03)
    cerrar(tar)
    print(f"    eventos con alarma: {n_marca} de {len(qe_a)}")
    print("    Fin de la fase 5.")


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 6 — ¿POR QUÉ ELEGIR UN SOM? (4 mini-validaciones)
# ══════════════════════════════════════════════════════════════════════════════
def fase6():
    aviso("FASE 6 · ¿Por qué elegir un SOM?   (sin etiquetas · visual · "
          "flexible · robusto)")
    fig, ax0 = figura_presentation()
    titulo_slide(ax0, "¿Por qué elegir un SOM?",
                 "Cuatro ventajas, probadas una a una. Cada mini-experimento "
                 "se entrena AQUÍ, en segundos.")
    gs = GridSpec(2, 2, figure=fig, left=.055, right=.945, top=.66,
                  bottom=.07, wspace=.28, hspace=.60)

    tarjetas_spec = [
        ("01  SIN ETIQUETAS", "dos grupos sin etiquetas → dos regiones\n"
                              "diferenciadas de la malla.", TURQUESA),
        ("02  VISUAL", "la MATRIZ-U: un «mapa topográfico»\n"
                       "de la red, siempre disponible.", LILA),
        ("03  FLEXIBLE", "círculo, espiral o cuadrado: la MISMA\n"
                         "red se adapta a cualquier forma.", DORADO),
        ("04  ROBUSTO", "con 15% de datos perdidos + ruido,\n"
                        "los 3 grupos siguen separados.", CORAL),
    ]
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    # rótulos de los 4 cuadrantes sobre ax0 (fuera de los mini-gráficos)
    quads = [(.055, .704), (.554, .704), (.055, .352), (.554, .352)]
    for (qx, qy), (t, d, c) in zip(quads, tarjetas_spec):
        ax0.text(qx, qy, t, fontsize=14.5, color=c, fontweight="bold",
                 va="top")
        ax0.text(qx, qy - .025, d, fontsize=9, color=GRIS, va="top")
    for ax in axes:
        ax.set_facecolor(AZUL)
        panel_oscuro(ax, None)

    rng = np.random.default_rng(55)

    # ── 1·SIN ETIQUETAS: la estructura 'dos lunas' emerge sola ─────────────
    def datos_dos_lunas(n, ruido=.045):
        """Dos medias-lunas clásicas, dentro del cuadrado [0,1]² limpio."""
        a = np.linspace(0, np.pi, n // 2)
        c, s = np.cos(a), np.sin(a)
        A = np.c_[(c + 1) / 3, (s + .5) / 1.5]
        B = np.c_[(2 - c) / 3, (1 - s) / 1.5]
        P = np.vstack([A, B]) + rng.normal(0, ruido, (n, 2))
        return np.clip(P, 0, 1)

    X1 = datos_dos_lunas(190)
    ids1 = np.array([0] * 95 + [1] * 95)     # (solo para COLOREAR de chequeo;
    #                                          el SOM nunca lo recibe)
    s1 = SOM(8, 8, 2, semilla=50)
    s1.entrenar(X1, epocas=60, t0=.4, r0=4.0, rng=np.random.default_rng(3))
    bms = [tuple(s1.bmu_de(x)[0]) for x in X1]
    hit = {}
    for b, x, cid in zip(bms, X1, ids1):
        hit.setdefault(b, []).append((x, cid))
    cmapa = LinearSegmentedColormap.from_list("x", ["#eaeaea", "#9aaac2"])
    axes[0].imshow(s1.matriz_u(), cmap=cmapa, alpha=.30,
                   extent=[-.5, 7.5, 7.5, -.5])
    axes[0].scatter(X1[:, 0] * 7, X1[:, 1] * 7,
                    c=[PALETA_TOP[i] for i in ids1], s=13, alpha=.9, lw=0)

    # ── 2·VISUAL: matriz-U como mapa de calor del «territorio» ─────────────
    axes[1].imshow(s1.matriz_u(), cmap=UM_CMAP, extent=[-.5, 7.5, 7.5, -.5])
    for b, pts in hit.items():
        if len(pts) >= 4:
            cid = max(set(k for _, k in pts), key=lambda k:
                      sum(1 for _, kk in pts if kk == k))
            axes[1].scatter(b[1], b[0], s=210, marker="o",
                            color=PALETA_TOP[cid], alpha=.30, lw=0, zorder=5)

    # ── 3·FLEXIBLE: tres formas geométricas muy distintas, misma red ───────
    conj = {}
    th = np.linspace(0, 2 * np.pi, 70)
    circ_0 = np.c_[.5 + .42 * np.cos(th), .5 + .42 * np.sin(th)]
    conj["círculo"] = np.clip(circ_0 + rng.normal(0, .05, (70, 2)), 0, 1)
    conj["espiral"] = np.clip(
        np.c_[.5 + (.45 * th / (2 * np.pi)) * np.cos(3 * th),
              .5 + (.45 * th / (2 * np.pi)) * np.sin(3 * th)]
        + rng.normal(0, .02, (70, 2)), 0, 1)
    conj["cuadrado"] = np.clip(
        np.c_[rng.uniform(.08, .93, (70,)), rng.choice([.08, .93], (70,))]
        + rng.normal(0, .02, (70, 2)), 0, 1)
    for k, (nom, D) in enumerate(conj.items()):
        Dn = SOM.normalizar(D)
        Ss = SOM(7, 7, 2, semilla=9 + k)
        Ss.entrenar(Dn, epocas=55, t0=.35, r0=3.5, rng=np.random.default_rng(4))
        mini = fig.add_subplot(3, 3, 3, zorder=0) if False else None
        ins = axes[2].inset_axes([.02 + k * .345, .02, .30, .74])
        ins.imshow(Ss.matriz_u(), cmap=UM_CMAP, extent=[-.5, 6.5, 6.5, -.5])
        ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_color("#2b3d63")
        ins.set_title(nom, color=CREMA, fontsize=8.6, pad=2)
    axes[2].axis("off")

    # ── 4·ROBUSTO: ruido alto + 15% de rasgos perdidos → igual se ordena ───
    xb = np.vstack([np.clip(rng.normal(c, .09, (60, 2)), 0, 1)
                    for c in ([.3, .3], [.7, .3], [.5, .75])])
    idx = np.array([0] * 60 + [1] * 60 + [2] * 60)   # (color de CHEQUEO)
    Xruido = np.clip(xb + rng.normal(0, .25, xb.shape), 0, 1)
    hueco = rng.random(xb.shape) < .15          # 15% de rasgos «perdidos»
    Xinh = np.where(hueco, .5, Xruido)          # estrategia típica: rellenar
    s4 = SOM(8, 8, 2, semilla=70)
    s4.entrenar(Xinh, epocas=70, t0=.4, r0=4.0, rng=np.random.default_rng(14))
    um4 = s4.matriz_u()
    axes[3].imshow(um4, cmap=UM_CMAP, alpha=.62, extent=[-.5, 7.5, 7.5, -.5])
    axes[3].scatter(xb[:, 0] * 7, xb[:, 1] * 7,
                    c=[PALETA_TOP[i] for i in idx], s=7, lw=0, alpha=.55)

    tar = Ventana(fig)
    while not tar.cerrada:
        tar.refrescar(0.03)
    cerrar(tar)
    print("    Fin de la fase 6.")


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 7 — EL MAPA TAMBIÉN EXIGE CRITERIO
# ══════════════════════════════════════════════════════════════════════════════
def fase7():
    aviso("FASE 7 · El mapa también exige criterio")
    fig, ax0 = figura_presentation()
    titulo_slide(ax0, "El mapa también exige criterio",
                 "Cuatro tensiones que conviene gestionar desde el diseño "
                 "experimental.")
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1], left=.04, right=.96,
                  top=.64, bottom=.31, wspace=.30, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax in axes:
        ax.set_facecolor(AZUL)

    datos = datos_colores(n=250, semilla=13)
    Xn = SOM.normalizar(datos)

    configuraciones = [
        ("5×5  · MUY PEQUEÑO", 5, 5, "todo queda APRETADO: grupos distintos\n"
                                     "terminan compartiendo la misma casilla", CORAL),
        ("10×10 · EQUILIBRADO", 10, 10, "grupos bien separados, buen\n"
                                        "detalle sin huecos vacíos", VERDE),
        ("22×22 · MUY GRANDE", 22, 22, "con pocos datos quedan HUECOS;\n"
                                       "más lento y fácil de sobreajustar", LILA),
    ]
    for ax, (t, f, c, nota, col) in zip(axes, configuraciones):
        Ss = SOM(f, c, 3, semilla=45)
        Ss.entrenar(Xn, epocas=55, t0=.5, r0=max(f, c) / 2,
                    rng=np.random.default_rng(16))
        zz = Ss.matriz_u()
        ax.imshow(zz, cmap=UM_CMAP, extent=[-.5, c - .5, f - .5, -.5])
        panel_oscuro(ax, t)
        ax.text(.5, -.085, nota, color=col, fontsize=9.5,
                transform=ax.transAxes, va="top", ha="center")

    # las 4 tensiones de la diapositiva, como chips (2×2 sobre el fondo)
    ax0.text(.045, .238, "LAS CUATRO TENSIONES A GESTIONAR", color=DORADO,
             fontsize=13, fontweight="bold", va="top")
    chips = [("Tamaño del mapa y tasa de aprendizaje", TURQUESA),
             ("Interpretación (exige experiencia)", LILA),
             ("Escalabilidad en volúmenes extremos", DORADO),
             ("Heterogeneidad y sobreajuste", CORAL)]
    for k, (tt, c) in enumerate(chips):
        x = .045 + (k % 2) * .475
        y = .148 - (k // 2) * .068
        tarjeta(ax0, x, y, .44, .062, c)
        ax0.text(x + .22, y + .031, tt, color=CREMA, fontsize=10,
                 ha="center", va="center", fontweight="bold")

    tarjeta(ax0, .14, .022, .72, .052, TURQUESA)
    ax0.text(.50, .048, "No hay configuración universal:  VALIDAR · "
                        "COMPARAR · VOLVER A ENTRENAR",
             color=CREMA, fontsize=12.5, ha="center", va="center",
             fontweight="bold")

    tar = Ventana(fig)
    while not tar.cerrada:
        tar.refrescar(0.03)
    cerrar(tar)
    print("    Fin de la fase 7.")


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 8 — EL SOM DEL FUTURO NO ESTARÁ SOLO (mapa vivo en tiempo real)
# ══════════════════════════════════════════════════════════════════════════════
def fase8():
    aviso("FASE 8 · El SOM del futuro: flujos en tiempo real y adaptación")
    fig, ax0 = figura_presentation()
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1],
                  left=.045, right=.955, top=.80, bottom=.08, wspace=.20)
    axmap = fig.add_subplot(gs[0])
    axtxt = fig.add_subplot(gs[1])
    for ax in (axmap, axtxt):
        ax.set_facecolor(AZUL)
    axtxt.axis("off")

    titulo_slide(ax0, "El SOM del futuro no estará solo",
                 "IA híbrida · tiempo real · adaptación: el mapa como capa "
                 "viva para entender sistemas cambiantes.")

    # ---- mapa vivo --------------------------------------------------------
    panel_oscuro(axmap, "MAPA VIVO · se está ADAPTANDO a un flujo que cambia")
    som = SOM(10, 10, 3, semilla=90)
    rng = np.random.default_rng(91)
    CA, CB = np.array([.28, .30, .75]), np.array([.78, .70, .30])
    for _ in range(320):                                   # pre-entrena
        x = CA + rng.normal(0, .08, 3) if rng.random() < .5 \
            else CB + rng.normal(0, .08, 3)
        x = np.clip(x, 0, 1)
        som.paso(x, eta=.28, sigma=2.4)
    cua = celdas_mapa(axmap, som)

    N = 190
    golpes = np.empty((0, 2))            # BMUs de 2 dims (i, j) del flujo
    marcaA, = axmap.plot([], [], "o", ms=9, mfc="none", mec=DORADO,
                         mew=2.2, ls="", zorder=6)

    axtxt.set_xlim(0, 1); axtxt.set_ylim(0, 1)
    axtxt.text(.5, .95, " ese es un mapa EN TIEMPO REAL ", color=DORADO,
               fontsize=12.5, ha="center", fontweight="bold", va="top")
    axtxt.text(.03, .87,
               "   · llega un FLUJO de eventos, uno a uno.\n"
               "   · cada evento entrena al mapa UNA vez.\n"
               "   · a mitad del flujo, la realidad CAMBIA\n"
               "      (los temas se desplazan).",
               color=CREMA, fontsize=10.5, va="top")
    status = axtxt.text(.03, .58, "", color=TURQUESA, fontsize=11,
                        va="top", fontweight="bold")

    tar = Ventana(fig)
    for t in range(N):
        if tar.cerrada:
            break
        if t > N * 0.42:            # ¡drift! los centros empiezan a moverse
            CA = np.clip(CA + np.array([.0016, .0026, -.0018]), 0, 1)
            CB = np.clip(CB + np.array([-.0021, -.0017, .0024]), 0, 1)
        x = CA + rng.normal(0, .07, 3) if rng.random() < .5 \
            else CB + rng.normal(0, .07, 3)
        x = np.clip(x, 0, 1)
        eta = max(.05, .16 * (1 - t / N))
        sig = max(.8, 1.7 * (1 - t / N))
        bmu, d, G = som.paso(x, eta, sig)
        bmu = np.array(bmu)
        golpes = np.vstack([golpes, bmu + 0.46 + rng.normal(0, .12, 2)])
        # refresco cada pocos eventos
        if t % 3 == 0:
            for idx, r in enumerate(cua):
                i, j = divmod(idx, som.columnas)
                r.set_facecolor(np.clip(som.W[i, j], 0, 1))
            marcaA.set_data(golpes[-40:, 1], golpes[-40:, 0])
            axmap.set_title(
                "MAPA VIVO · t = %3d   η = %.3f   σ = %.2f%s" %
                (t, eta, sig,
                 "   ▸ ¡DRIFT! adaptándose…"
                 if t > N * .42 else ""),
                color=(CORAL if t > N * .42 else CREMA), fontsize=10,
                pad=10, fontweight="bold")
            tar.refrescar(0.004)
    # estado final de la vista
    for idx, r in enumerate(cua):
        i, j = divmod(idx, som.columnas)
        r.set_facecolor(np.clip(som.W[i, j], 0, 1))
    axmap.set_title("MAPA VIVO · al final del flujo: ya se ADAPTÓ al cambio",
                    color=CREMA, fontsize=10.5, pad=10, fontweight="bold")
    axtxt.text(.03, .38, "  el mapa «siguió» al cambio de tendencia\n"
                         "  sin reentrenar desde cero.",
               color=VERDE, fontsize=11, va="top", fontweight="bold")
    tarjeta(axtxt, .04, .24, .90, .08, VERDE)
    axtxt.text(.50, .28, "sistemas que APRENDEN mientras SIRVEN",
               color=CREMA, fontsize=11, ha="center", va="center",
               fontweight="bold")

    while not tar.cerrada:
        tar.refrescar(0.02)
    cerrar(tar)
    print("    Fin de la fase 8.")


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 9 — CIERRE: ORDENA · REVELA · INVITA
# ══════════════════════════════════════════════════════════════════════════════
def fase9():
    aviso("FASE 9 · Cierre")
    fig, ax0 = figura_presentation()
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1); ax0.axis("off")

    ax0.text(.5, .93, "ORDENA datos complejos", color=TURQUESA, fontsize=23,
             ha="center", fontweight="bold")
    ax0.text(.5, .885, "REVELA patrones útiles", color=DORADO, fontsize=23,
             ha="center", fontweight="bold")
    ax0.text(.5, .84, "INVITA a experimentar", color=LILA, fontsize=23,
             ha="center", fontweight="bold")

    # triángulo central
    centro = (.5, .555)
    pts = [(.5, .745), (.30, .415), (.70, .415)]
    for (a, b) in ((0, 1), (1, 2), (2, 0)):
        ax0.add_patch(FancyArrowPatch(pts[a], pts[b],
                     arrowstyle="-|>", mutation_scale=26, lw=2.6,
                     color=GRIS, alpha=.85, connectionstyle="arc3,rad=-0.16"))
    ax0.text(centro[0], centro[1] + .05, "SOM", color=CREMA, fontsize=30,
             ha="center", fontweight="bold")
    ax0.text(centro[0], centro[1] - .028, "una herramienta práctica para\n"
             "convertir complejidad en territorio.",
             color=GRIS, fontsize=11, ha="center", va="top", style="italic")

    ax0.text(.5, .20, "¿Preguntas?", color=CREMA, fontsize=26, ha="center",
             fontweight="bold")
    ax0.text(.5, .135, "Reflexionemos sobre el potencial de los SOM para "
             "transformar el análisis de datos complejos.",
             color=GRIS, fontsize=12, ha="center")

    tarjeta(ax0, .18, .045, .64, .052, DORADO)
    ax0.text(.5, .071, "GRUPO 10 · Oscar Segovia · Martha Diaz · "
             "Brandon Garcia · Juan Villalobos",
             color=CREMA, fontsize=10.5, ha="center", va="center",
             fontweight="bold")

    tar = Ventana(fig)
    while not tar.cerrada:
        tar.refrescar(0.03)
    cerrar(tar)
    print("    Fin de la fase 9.")


# ══════════════════════════════════════════════════════════════════════════════
#  M A I N
# ══════════════════════════════════════════════════════════════════════════════
def principal():
    print("\n\033[1;36m" + "=" * 74)
    print("  MAPAS AUTO-ORGANIZADOS DE KOHONEN (SOM) — DEMOSTRACIÓN EN PYTHON")
    print("=" * 74 + "\033[0m")
    print("  9 fases visuales. Cada una abre una ventana: watch, aprende,"
          " ciérrala,")
    print("  pulsa ENTER y continúa con la siguiente.\n")

    try:                                        # solo imprime 9 si está 0..9
        pasos = [fase1, fase2, fase3, fase4, fase5, fase6, fase7, fase8,
                 fase9]
        for k, fn in enumerate(pasos, 1):
            try:
                fn()
            except Exception as exc:
                print(f"    \033[31m(fase {k} interrumpida: {exc})\033[0m")
    except KeyboardInterrupt:
        print("\n    ¡Demo detenida por el usuario! Hasta pronto 👋")

    plt.close("all")
    print("\n\033[1;36m  ═══ DEMO TERMINADA — ¡cada nodo de la red ya dorme! "
          "═══\033[0m\n")


if __name__ == "__main__":
    principal()
