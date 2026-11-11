#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
════════════════════════════════════════════════════════════════════════════
  E J E R C I C I O   N U M É R I C O   ·   S O M   D E   K O H O N E N
  "Mapas Auto-Organizados (SOM): Redes que Ordenan Solas" — GRUPO 10
════════════════════════════════════════════════════════════════════════════

La demo ANTIGUA (visual) quedó en demostracion_som.py.
Esta NUEVA resuelve un EJERCICIO NUMÉRICO completo, paso a paso, donde
CADA fórmula del algoritmo se aplica con números visibles:

  ▸ 4 datos en 2 dimensiones, mapa de 2×2 neuronas (seguible a mano).
  ▸ PASO 1 ENTRADA     → se presenta el dato (normalizado).
  ▸ PASO 2 COMPETENCIA → d(x,W) = √( (x₁−W₁)² + (x₂−W₂)² )  para TODAS
                         las neuronas; gana la menor → BMU.
  ▸ PASO 3 COOPERACIÓN → h = exp( − d(r, r_BMU)² / (2σ²) )  para cada
                         neurona; y ADAPTACIÓN W(nuevo) = W + η·h·(x−W).
  ▸ PASO 4 CONVERGENCIA→ η(t) = η₀·(1−t/T)  y  σ(t) = σ₀·(1−t/T).

INTERACTIVO: botón ▸ (o ESPACIO / →) avanza UN paso mostrando la fórmula
general y los números sustituidos; ⏩ AUTO lo reproduce solo; ⟲ reinicia.
Todo queda también impreso en la terminal (útil para el informe).

Ejecutar:   python3 ejercicio_numerico_som.py
"""

import numpy as np
import matplotlib


def _elegir_backend():
    for b in ("GTK3Agg", "TkAgg", "QtAgg", "MacOSX", "Agg"):
        try:
            matplotlib.use(b)
            return b
        except Exception:
            continue
    matplotlib.use("Agg")
    return "Agg"


BACKEND = _elegir_backend()
GUI_OK = BACKEND != "Agg"

import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, Circle, Rectangle,
                                FancyArrowPatch)
from matplotlib.widgets import Button

plt.rcParams["font.family"] = "DejaVu Sans"

# ── paleta (estética de la presentación) ────────────────────────────────────
AZUL    = "#16233f"
AZUL_2  = "#1f3357"
CREMA   = "#f5f1e8"
DORADO  = "#e8b64a"
TURQUESA= "#2ec4b6"
CORAL   = "#ef476f"
LILA    = "#b388eb"
VERDE   = "#7dd463"
GRIS    = "#8a93a5"

# ════════════════════════════════════════════════════════════════════════════
#  EL EJERCICIO — datos, pesos iniciales y parámetros (todos a mano)
# ════════════════════════════════════════════════════════════════════════════
DATOS  = [np.array([.2, .8]), np.array([.3, .9]),
          np.array([.8, .2]), np.array([.9, .3])]
NOMB   = ["X₁", "X₂", "X₃", "X₄"]
GRUPO  = [0, 0, 1, 1]                       # A, A, B, B (solo para COLOREAR;
GRUPO_L= ["A", "B"]                         #  el SOM jamás recibe esto)
C_GRUPO= [TURQUESA, CORAL]

W_INI  = np.array([[[.5, .5], [.1, .2]],
                   [[.9, .8], [.4, .1]]])   # W[i][j] = (w1, w2)
ETIQ   = np.array([["W₁₁", "W₁₂"], ["W₂₁", "W₂₂"]])
POS    = np.zeros((2, 2, 2), dtype=float)   # coordenadas (i,j) en la malla
for i in range(2):
    for j in range(2):
        POS[i, j] = (i, j)

EPOCAS, N_DAT = 4, 4
T_TOT  = EPOCAS * N_DAT                     # 16 pasos de entrenamiento
ETA0, SIG0 = .60, 1.50
ETA_MIN, SIG_MIN = .02, .35

F, C = 2, 2                                 # filas × columnas del mapa


def f3(v):  return f"{v:.3f}"
def fp(v):  return f"({v[0]:.3f}, {v[1]:.3f})"


# ════════════════════════════════════════════════════════════════════════════
#  utilidades de presentación
# ════════════════════════════════════════════════════════════════════════════
def aviso(t):
    print("\n  \033[96m▸\033[0m " + t)


def figura(w=15.5, h=9.0):
    fig = plt.figure(figsize=(w, h), facecolor=AZUL)
    ax0 = fig.add_axes([0, 0, 1, 1], zorder=-20)
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1); ax0.axis("off")
    grad = np.zeros((256, 1, 3))
    arriba = np.array([0x0e/255, 0x16/255, 0x28/255])
    abajo  = np.array([0x1d/255, 0x2f/255, 0x52/255])
    for k in range(3):
        grad[:, 0, k] = np.linspace(arriba[k], abajo[k], 256)
    ax0.imshow(grad, extent=(0, 1, 0, 1), aspect="auto", zorder=-21)
    ax0.add_patch(FancyBboxPatch(
        (0.010, 0.010), 0.980, 0.980,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        fc="none", ec=DORADO, lw=2.4, alpha=.40, zorder=-20))
    ax0.text(.985, .022, "EJERCICIO NUMÉRICO · SOM de Kohonen · GRUPO 10",
             fontsize=8.5, color=GRIS, ha="right", va="bottom", zorder=-20)
    return fig, ax0


def titulo_ventana(fig, t):
    try:
        fig.canvas.manager.set_window_title(t)
    except Exception:
        pass


def tarjeta(ax, x, y, w, h, color, lw=2.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.022",
        fc=AZUL, ec=color, lw=lw))


def panel_oscuro(ax, titulo=None):
    ax.set_facecolor("#141f38")
    for sp in ax.spines.values():
        sp.set_color("#2b3d63")
    ax.tick_params(colors=GRIS, labelsize=9)
    if titulo:
        ax.set_title(titulo, color=CREMA, fontsize=11.5, pad=10,
                     fontweight="bold")


class Ventana:
    def __init__(self, fig):
        self.fig = fig
        self.cerrada = not GUI_OK
        fig.canvas.mpl_connect("close_event",
                               lambda e: setattr(self, "cerrada", True))
        plt.show(block=False)

    def refrescar(self, pausa=.001):
        if self.cerrada:
            return False
        try:
            self.fig.canvas.draw_idle()
            plt.pause(pausa)
            return True
        except Exception:
            self.cerrada = True
            return False


def cerrar(v):
    try:
        plt.close(v.fig)
    except Exception:
        pass


def boton(fig, rect, txt, color, cb, fs=10.5):
    bax = fig.add_axes(rect)
    bax.set_facecolor(AZUL_2)
    b = Button(bax, txt, color=AZUL_2, hovercolor="#31497c")
    b.label.set_color(CREMA); b.label.set_fontsize(fs)
    for sp in bax.spines.values():
        sp.set_color(color); sp.set_linewidth(2.0)
    bax.zorder = 40
    b.on_clicked(cb)
    return b


# ════════════════════════════════════════════════════════════════════════════
#  VISTAS — mapa 2×2, espacio de datos y barras η/σ  (sirven a todas las fases)
# ════════════════════════════════════════════════════════════════════════════
class MapaVista:
    """Cuadrícula 2×2: cada tarjeta muestra su vector de pesos W = (w₁, w₂)."""

    def __init__(self, ax):
        self.ax = ax
        ax.set_xlim(-.15, 2.15); ax.set_ylim(2.15, -.15)   # fila 0 arriba
        ax.set_aspect("equal"); ax.axis("off")
        self.cards, self.t_w, self.t_v = [], [], []
        self.hrects, self.dots = [], []
        for i in range(F):
            fila_c, fila_t, fila_v, fila_h, fila_d = [], [], [], [], []
            for j in range(C):
                c = FancyBboxPatch((j + .04, i + .04), .92, .92,
                                   boxstyle="round,pad=0.01,rounding_size=0.06",
                                   fc="#17244a", ec="#31497c", lw=2.0)
                ax.add_patch(c); fila_c.append(c)
                fila_h.append(Rectangle((j, i), 1, 1, fc=LILA, alpha=0,
                                        ec="none", zorder=1))
                ax.add_patch(fila_h[-1])
                fila_t.append(ax.text(j + .5, i + .30, ETIQ[i, j], color=CREMA,
                                      fontsize=15, ha="center",
                                      fontweight="bold", zorder=3))
                fila_v.append(ax.text(j + .5, i + .62, "", color=DORADO,
                                      fontsize=11, ha="center", zorder=3))
                fila_d.append([])
            self.cards.append(fila_c); self.t_w.append(fila_t)
            self.t_v.append(fila_v); self.hrects.append(fila_h)
            self.dots.append(fila_d)
        self.star = ax.text(-1, -1, "★", color=DORADO, fontsize=34,
                            ha="center", va="center", alpha=0, zorder=5)

    def actualizar(self, W, bmu=None, hs=None, miembros=None):
        for i in range(F):
            for j in range(C):
                self.t_v[i][j].set_text(f"( {W[i, j, 0]:.3f}, {W[i, j, 1]:.3f} )")
                self.cards[i][j].set_edgecolor(
                    DORADO if bmu is not None and (i, j) == tuple(bmu)
                    else "#31497c")
                self.cards[i][j].set_linewidth(
                    3.4 if bmu is not None and (i, j) == tuple(bmu) else 2.0)
                h = hs[i, j] if hs is not None else None
                self.hrects[i][j].set_alpha(0 if h is None else .55 * h)
                for d in self.dots[i][j]:
                    d.remove()
                self.dots[i][j] = []
                if miembros:
                    ms = miembros.get((i, j), [])
                    for k, (_, g) in enumerate(ms[:4]):
                        d = Circle((j + .28 + k * .15, i + .87), .05,
                                   fc=C_GRUPO[g], ec="none", zorder=4)
                        self.ax.add_patch(d)
                        self.dots[i][j].append(d)
        if bmu is None:
            self.star.set_alpha(0)
        else:
            self.star.set_position((bmu[1] + .84, bmu[0] + .17))
            self.star.set_alpha(1)


class EspacioVista:
    """El espacio de datos [0,1]²: los datos (círculos) y los pesos de las
    neuronas (cuadrados dorados) MIGRANDO hacia los datos."""

    def __init__(self, ax):
        self.ax = ax
        ax.set_xlim(-.10, 1.10); ax.set_ylim(-.10, 1.10)
        ax.set_aspect("equal")
        panel_oscuro(ax, "ESPACIO DE DATOS · los pesos (■) migran")
        ax.set_xlabel("x₁", color=GRIS); ax.set_ylabel("x₂", color=GRIS)
        ax.grid(color="#233557", lw=.8, alpha=.7)
        # datos
        for k, x in enumerate(DATOS):
            ax.scatter([x[0]], [x[1]], s=300, c=C_GRUPO[GRUPO[k]],
                       ec="#0e172c", lw=1.4, zorder=5)
            if k < 2:      # X₁, X₂: etiqueta ARRIBA (allí viven los pesos)
                lx, ly, va = x[0] - .02, x[1] + .055, "bottom"
            else:          # X₃, X₄: etiqueta DEBAJO
                lx, ly, va = x[0] - .02, x[1] - .06, "top"
            ax.text(lx, ly, f"{NOMB[k]} = ({x[0]:.1f}, {x[1]:.1f})",
                    color=C_GRUPO[GRUPO[k]], fontsize=9.5, fontweight="bold",
                    va=va, zorder=6)
        ax.text(.02, .965, "grupo A (azul) y B (coral): SOLO colores para ti —"
                           " el SOM no los ve", color=GRIS, fontsize=8.2,
                style="italic", va="top")
        # pesos
        self.wpts = ax.scatter([], [], s=300, marker="s", c=DORADO,
                               ec="#fff", lw=1.2, zorder=6)
        self.wlbl = [ax.text(0, 0, "", color=CREMA, fontsize=9, zorder=7,
                             bbox=dict(fc="#0e172c", ec="none", alpha=.85,
                                       boxstyle="round,pad=.22")) for _ in range(4)]
        self.dat_ring = Circle((-.5, -.5), .055, fc="none", ec=CREMA,
                               lw=2.4, ls="--", alpha=0, zorder=7)
        self.bmu_ring = Circle((-.5, -.5), .058, fc="none", ec=DORADO,
                               lw=2.6, alpha=0, zorder=7)
        ax.add_patch(self.dat_ring); ax.add_patch(self.bmu_ring)
        self.flechas = []

    def actualizar(self, W, prev=None, bmu=None, dato=None):
        pts = [tuple(W[i, j]) for i in range(F) for j in range(C)]
        self.wpts.set_offsets(pts)
        for k, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            self.wlbl[k].set_position((W[i, j, 0] + .05, W[i, j, 1] + .038))
            self.wlbl[k].set_text(ETIQ[i, j])
        for f in self.flechas:
            f.remove()
        self.flechas = []
        if prev is not None:
            for i in range(F):
                for j in range(C):
                    a, b = prev[i, j], W[i, j]
                    if np.linalg.norm(b - a) > .012:
                        fr = FancyArrowPatch(tuple(a), tuple(b),
                                             arrowstyle="-|>", lw=2.0,
                                             mutation_scale=16, color=DORADO,
                                             alpha=.90, zorder=5)
                        self.ax.add_patch(fr)
                        self.flechas.append(fr)
        if dato is None:
            self.dat_ring.set_alpha(0)
        else:
            self.dat_ring.center = tuple(dato)
            self.dat_ring.set_alpha(1)
        if bmu is None:
            self.bmu_ring.set_alpha(0)
        else:
            self.bmu_ring.center = tuple(W[tuple(bmu)])
            self.bmu_ring.set_alpha(1)


class BarrasVista:
    """Barras η y σ con su valor numérico y t/T."""

    def __init__(self, ax):
        self.ax = ax
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        self.t_txt = ax.text(.5, .99, "", color=GRIS, fontsize=8.5,
                             ha="center", va="top", fontweight="bold")
        self.barras = []
        for y, c, nombre in ((.56, CORAL, "η  tasa"),
                             (.13, LILA, "σ  radio")):
            ax.add_patch(FancyBboxPatch((.02, y), .96, .26,
                         boxstyle="round,pad=0.004,rounding_size=0.02",
                         fc="#0e172c", ec="#2b3d63", lw=1))
            fg = FancyBboxPatch((.27, y + .05), .001, .16,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                fc=c, ec="none")
            ax.add_patch(fg)
            ax.text(.045, y + .13, nombre, color=c, fontsize=8.5,
                    va="center", fontweight="bold")
            tx = ax.text(.955, y + .13, "", color=c, fontsize=9, ha="right",
                         va="center", fontweight="bold")
            self.barras.append((fg, tx))

    def actualizar(self, eta, sig, t, T):
        self.t_txt.set_text(f"t = {t}  de  T = {T}  pasos totales")
        vals = (eta, sig); tops = (ETA0, SIG0)
        for (fg, tx), v, top in zip(self.barras, vals, tops):
            fg.set_width(max(.002, .55 * v / top))
            tx.set_text(f"{v:.3f}")


# ════════════════════════════════════════════════════════════════════════════
#  EL EJERCICIO GUIADO — máquina de estados paso a paso
# ════════════════════════════════════════════════════════════════════════════
PASOS = ["ENTRADA", "COMPETENCIA", "COOPERACIÓN", "ADAPTACIÓN", "CONVERGENCIA"]
PASOS_COLOR = [TURQUESA, CORAL, LILA, DORADO, VERDE]


class Ejercicio:
    def __init__(self, fig, axmap, axbars, axsp, axcal, ax0):
        self.fig = fig
        self.mapa = MapaVista(axmap)
        self.esp = EspacioVista(axsp)
        self.bars = BarrasVista(axbars)
        axcal.axis("off")
        axcal.set_xlim(0, 1); axcal.set_ylim(0, 1)
        self.chip = axcal.text(.5, .965, "", color=CREMA, fontsize=13.5,
                               ha="center", va="top", fontweight="bold")
        self.lineas = axcal.text(.035, .875, "", color=CREMA, fontsize=9.5,
                                 va="top", linespacing=1.85)
        self.hud = fig.text(.5, .088, "", color=CREMA, fontsize=11,
                            ha="center", fontweight="bold")
        self.auto = False
        self.salir = False
        boton(fig, [.045, .022, .175, .058], "▸  SIGUIENTE PASO",
              TURQUESA, lambda e: self.siguiente())
        self.bauto = boton(fig, [.25, .022, .125, .058], "⏩  AUTO",
                           LILA, self._alternar_auto)
        boton(fig, [.405, .022, .15, .058], "⟲  REINICIAR",
              DORADO, lambda e: self.reiniciar())
        boton(fig, [.585, .022, .115, .058], "✕  TERMINAR",
              CORAL, lambda e: setattr(self, "salir", True))
        fig.canvas.mpl_connect("key_press_event", self._tecla)
        self.reiniciar()

    # ── controles ────────────────────────────────────────────────────────
    def _alternar_auto(self, _):
        self.auto = not self.auto
        self.bauto.label.set_text("⏸  PAUSA" if self.auto else "⏩  AUTO")

    def _tecla(self, e):
        if e.key in ("right", " "):
            self.siguiente()
        elif e.key == "a":
            self._alternar_auto(None)
        elif e.key == "r":
            self.reiniciar()

    # ── estado ───────────────────────────────────────────────────────────
    def reiniciar(self):
        self.W = W_INI.copy()
        self.prevW = W_INI.copy()
        self.ep, self.di, self.paso, self.t = 0, 0, 0, 0
        self.eta, self.sig = ETA0, SIG0
        self.bmu = None
        self.hs = None
        self.miembros = {(i, j): [] for i in range(F) for j in range(C)}
        self.asign = {}
        self.fin = False
        self.auto = False
        self.bauto.label.set_text("⏩  AUTO")
        self.mapa.actualizar(self.W)
        self.esp.actualizar(self.W)
        self.bars.actualizar(self.eta, self.sig, 0, T_TOT)
        self.poner("DEMO PASO A PASO", [
            "4 datos · 2 dimensiones · mapa de 2×2 neuronas",
            "",
            "Pulsa  ▸ SIGUIENTE PASO  (o la tecla ESPACIO / →)",
            "para aplicar CADA fórmula del SOM con números.",
            "",
            "⏩ AUTO reproduce todos los pasos solo.",
            "⟲ REINICIAR empieza de cero."], GRIS)
        self._hud()

    def _hud(self):
        if self.fin:
            self.hud.set_text("✔ EJERCICIO TERMINADO  ·  ⟲ reiniciar para "
                              "volver a empezar")
            self.hud.set_color(VERDE)
        else:
            self.hud.set_text(
                f"ÉPOCA {self.ep+1}/{EPOCAS}  ·  dato {NOMB[self.di]} "
                f"({self.di+1}/{N_DAT})  ·  PASO {self.paso+1}/5 · "
                f"t = {self.t}  ·  η = {self.eta:.3f}  ·  σ = {self.sig:.3f}")
            self.hud.set_color(CREMA)

    def poner(self, chip, lineas, color):
        self.chip.set_text(chip)
        self.chip.set_color(color)
        self.lineas.set_text("\n".join(lineas))

    # ── un paso ──────────────────────────────────────────────────────────
    def siguiente(self):
        if self.fin:
            return
        x = DATOS[self.di]
        p = self.paso

        if p == 0:                                            # 1 · ENTRADA
            lineas = [
                "Presentamos el dato de entrada:",
                "",
                f"   x  =  ( {x[0]:.3f}, {x[1]:.3f} )      ← dato {NOMB[self.di]}",
                "",
                f"   (grupo real: {GRUPO_L[GRUPO[self.di]]} — solo color; "
                "el SOM NO lo ve)",
                "",
                "La normalización ya está hecha:  x ∈ [0,1]²  ✓",
                f"   t = {self.t}   ·   η = {self.eta:.3f}   ·   σ = {self.sig:.3f}",
            ]
            self.poner(f"PASO 1/5 · ENTRADA", lineas, PASOS_COLOR[0])
            self.mapa.actualizar(self.W)
            self.esp.actualizar(self.W, dato=x)
            self._imprimir(p, lineas)

        elif p == 1:                                       # 2 · COMPETENCIA
            D = np.linalg.norm(self.W - x, axis=-1)
            self.bmu = tuple(np.unravel_index(np.argmin(D), D.shape))
            lineas = ["Cada neurona COMPITE: distancia euclídea al dato:",
                      "",
                      "   d(x, W) = √( (x₁−W₁)² + (x₂−W₂)² )", ""]
            for i in range(F):
                for j in range(C):
                    dif = x - self.W[i, j]
                    s1, s2 = dif[0]**2, dif[1]**2
                    marca = "   ★  ¡BMU!" if (i, j) == self.bmu else ""
                    lineas.append(
                        f"   d(x, {ETIQ[i, j]}) = √(({x[0]:.2f}−{self.W[i,j,0]:.2f})²"
                        f" + ({x[1]:.2f}−{self.W[i,j,1]:.2f})²)")
                    lineas.append(
                        f"               = √({s1:.3f} + {s2:.3f}) = {D[i,j]:.3f}{marca}")
            self.poner("PASO 2/5 · COMPETENCIA", lineas, PASOS_COLOR[1])
            self.mapa.actualizar(self.W, bmu=self.bmu)
            self.esp.actualizar(self.W, bmu=self.bmu, dato=x)
            self._imprimir(p, lineas)

        elif p == 2:                                       # 3 · COOPERACIÓN
            self.sig = max(SIG_MIN, SIG0 * (1 - self.t / T_TOT))
            dr = np.linalg.norm(POS - np.array(self.bmu), axis=-1)
            self.hs = np.exp(-(dr ** 2) / (2 * self.sig ** 2))
            lineas = [
                "La BMU NO aprende sola: su VECINDARIO coopera.",
                "",
                f"   σ({self.t}) = σ₀·(1 − t/T) = 1.500·(1 − {self.t}/{T_TOT}) "
                f"= {self.sig:.3f}",
                "   h = exp( − d(r, r_BMU)² / (2σ²) )",
                f"   (2σ² = {2*self.sig**2:.3f})", ""]
            for i in range(F):
                for j in range(C):
                    marca = "   ★ BMU" if (i, j) == self.bmu else ""
                    lineas.append(
                        f"   h_{i+1}{j+1} = exp( −{dr[i,j]**2:.3f} / "
                        f"{2*self.sig**2:.3f} ) = {self.hs[i,j]:.3f}{marca}")
            self.poner("PASO 3/5 · COOPERACIÓN", lineas, PASOS_COLOR[2])
            self.mapa.actualizar(self.W, bmu=self.bmu, hs=self.hs)
            self.esp.actualizar(self.W, bmu=self.bmu, dato=x)
            self._imprimir(p, lineas)

        elif p == 3:                                        # 3b · ADAPTACIÓN
            self.eta = max(ETA_MIN, ETA0 * (1 - self.t / T_TOT))
            self.prevW = self.W.copy()
            delta = self.eta * self.hs[..., None] * (x - self.W)
            self.W = self.W + delta
            self.miembros[self.bmu].append((self.di, GRUPO[self.di]))
            self.asign[self.di] = self.bmu
            lineas = [
                "TODAS las neuronas se acercan al dato, según su h:",
                "",
                "   W(nuevo) = W + η·h·(x − W)",
                f"   η({self.t}) = η₀·(1 − t/T) = 0.600·(1 − {self.t}/{T_TOT}) "
                f"= {self.eta:.3f}", ""]
            for i in range(F):
                for j in range(C):
                    dif = x - self.prevW[i, j]
                    dd = delta[i, j]
                    marca = "  ★" if (i, j) == self.bmu else ""
                    lineas.append(
                        f"   {ETIQ[i,j]}: x−W = ({dif[0]:+.3f}, {dif[1]:+.3f}) · "
                        f"η·h = {self.eta*self.hs[i,j]:.3f}")
                    lineas.append(
                        f"     ΔW = ({dd[0]:+.3f}, {dd[1]:+.3f}) → "
                        f"W = {fp(self.W[i,j])}{marca}")
            self.poner("PASO 4/5 · ADAPTACIÓN", lineas, PASOS_COLOR[3])
            self.mapa.actualizar(self.W, bmu=self.bmu, miembros=self.miembros)
            self.esp.actualizar(self.W, prev=self.prevW, bmu=self.bmu, dato=x)
            self.bars.actualizar(self.eta, self.sig, self.t, T_TOT)
            self._imprimir(p, lineas)

        elif p == 4:                                      # 4 · CONVERGENCIA
            t1 = self.t + 1
            eta1 = max(ETA_MIN, ETA0 * (1 - t1 / T_TOT))
            sig1 = max(SIG_MIN, SIG0 * (1 - t1 / T_TOT))
            lineas = [
                "Los parámetros DECAEN: el mapa se ESTABILIZA.",
                "",
                f"   η(t+1) = η₀·(1 − t/T) = 0.600·(1 − {t1}/{T_TOT}) "
                f"= {eta1:.3f}",
                f"   σ(t+1) = σ₀·(1 − t/T) = 1.500·(1 − {t1}/{T_TOT}) "
                f"= {sig1:.3f}",
                "",
                "   η y σ cada vez más pequeños →",
                "   al principio aprende GRUESO, al final AFINA.",
                ""]
            if self.di == N_DAT - 1:
                lineas.append(f"   ▸ fin de la ÉPOCA {self.ep+1}: "
                              "todos los datos ya pasaron una vez.")
            self.poner("PASO 5/5 · CONVERGENCIA", lineas, PASOS_COLOR[4])
            self._imprimir(p, lineas)
            self.t, self.eta, self.sig = t1, eta1, sig1
            self.di += 1
            if self.di == N_DAT:
                self.di = 0
                self.ep += 1
                if self.ep == EPOCAS:
                    self.fin = True
                    self.resumen()
                    return
            self.paso = 0
            self.bmu = None
            self.hs = None
            self.mapa.actualizar(self.W)
            self.esp.actualizar(self.W, dato=None)
            self.bars.actualizar(self.eta, self.sig, self.t, T_TOT)
            self.poner("SIGUIENTE DATO", [
                "   Dato procesado ✓  (los pesos ya se movieron)",
                "",
                f"   Ahora presentamos {NOMB[self.di]}…",
                "",
                "   Pulsa ▸ para el PASO 1/5 · ENTRADA."], TURQUESA)
            self._hud()
            return

        self.paso += 1
        self._hud()

    def _imprimir(self, p, lineas):
        print(f"\n  ── PASO {p+1}/5 · {PASOS[p]}  "
              f"(época {self.ep+1}, dato {NOMB[self.di]}, t={self.t}) ──")
        for s in lineas:
            if s.strip():
                print("    " + s)

    # ── resultado final ──────────────────────────────────────────────────
    def resumen(self):
        ok = tot = 0
        for (i, j), ms in self.miembros.items():
            if not ms:
                continue
            dom = max(set(g for _, g in ms),
                      key=lambda g: sum(1 for _, gg in ms if gg == g))
            for _, g in ms:
                tot += 1
                ok += (g == dom)
        pureza = ok / max(tot, 1)
        asign_txt = "   ".join(
            f"{NOMB[k]} → {ETIQ[tuple(self.asign[k])]}" for k in range(N_DAT))
        lineas = [
            f"   {asign_txt}",
            "",
            "   Cada neurona terminó especializada en UN grupo:",
            "   los puntos azules en unas casillas, los corales en otras.",
            "",
            f"   PUREZA de los grupos emergentes:  {pureza:.0%}  ✓",
            "",
            "   ¡El SOM ORDENÓ los datos SIN etiquetas!",
            "   (los colores solo los usa el espectador)",
        ]
        self.poner("RESULTADO DEL EJERCICIO", lineas, VERDE)
        self.mapa.actualizar(self.W, miembros=self.miembros)
        self.esp.actualizar(self.W)
        self._hud()
        aviso("RESULTADO FINAL DEL EJERCICIO NUMÉRICO")
        for k in range(N_DAT):
            b = tuple(self.asign[k])
            print(f"    {NOMB[k]} = {fp(DATOS[k])}  →  BMU {ETIQ[b]} "
                  f"(grupo real {GRUPO_L[GRUPO[k]]})")
        print(f"    pureza de los grupos emergentes = {pureza:.0%}")


def fase_ejercicio():
    aviso("EJERCICIO NUMÉRICO GUIADO — 4 datos · mapa 2×2 · todas las fórmulas")
    fig, ax0 = figura()
    titulo_ventana(fig, "SOM · Ejercicio numérico paso a paso")
    ax0.text(.046, .965, "EJERCICIO NUMÉRICO DEL SOM, FÓRMULA A FÓRMULA",
             fontsize=21, color=CREMA, fontweight="bold", va="top")
    ax0.text(.046, .925, "datos 2D · mapa 2×2 · η₀ = 0.6 · σ₀ = 1.5 · "
             "T = 16 pasos — pulsa ▸ y sigue los números",
             fontsize=11.5, color=DORADO, va="top")

    axmap = fig.add_axes([.045, .24, .245, .56])
    axmap.set_facecolor(AZUL)
    panel_oscuro(axmap, "MAPA 2×2 · pesos de cada neurona")
    axbars = fig.add_axes([.045, .13, .245, .09])
    axsp = fig.add_axes([.36, .17, .29, .63])
    axcal = fig.add_axes([.70, .16, .265, .66])
    axcal.set_facecolor("#101b33")
    axcal.add_patch(FancyBboxPatch(
        (.01, .01), .98, .98, transform=axcal.transAxes,
        boxstyle="round,pad=0.004,rounding_size=0.03",
        fc="#101b33", ec="#2b3d63", lw=1.6, zorder=-1))

    ej = Ejercicio(fig, axmap, axbars, axsp, axcal, ax0)
    fig._ej = ej                      # gancho para pruebas

    if not GUI_OK:                    # modo sin ventana: resolverlo completo
        for _ in range(5 * N_DAT * EPOCAS + 2):
            ej.siguiente()
        cerrar(Ventana(fig))
        return

    tar = Ventana(fig)
    n = 0
    while not tar.cerrada and not ej.salir:
        if ej.auto and not ej.fin:
            n += 1
            if n % 16 == 0:
                ej.siguiente()
        tar.refrescar(.03)
    cerrar(tar)
    print("    Fin del ejercicio guiado.")


# ════════════════════════════════════════════════════════════════════════════
#  FASE 2 — CONVERGENCIA AUTOMÁTICA: 24 épocas de ese mismo ejercicio
# ════════════════════════════════════════════════════════════════════════════
def fase_convergencia():
    aviso("CONVERGENCIA — el mismo ejercicio, 24 épocas seguidas (animado)")
    fig, ax0 = figura()
    titulo_ventana(fig, "SOM · Convergencia del ejercicio (24 épocas)")
    ax0.text(.046, .965, "¿Y SI LO DEJAMOS ENTRENAR MÁS?  24 ÉPOCAS",
             fontsize=21, color=CREMA, fontweight="bold", va="top")
    ax0.text(.046, .925, "los mismos 4 datos y los mismos pesos iniciales — "
             "observa cómo los ■ migran hacia los grupos",
             fontsize=11.5, color=DORADO, va="top")

    axsp = fig.add_axes([.06, .18, .36, .62])
    axmap = fig.add_axes([.55, .30, .26, .50])
    axmap.set_facecolor(AZUL)
    panel_oscuro(axmap, "MAPA 2×2")
    axbars = fig.add_axes([.55, .17, .26, .10])

    esp = EspacioVista(axsp)
    mapa = MapaVista(axmap)
    bars = BarrasVista(axbars)
    estado = ax0.text(.5, .868, "", color=CREMA, fontsize=11.5,
                      ha="center", fontweight="bold")

    W = W_INI.copy()
    T = 24 * N_DAT
    miembros = {(i, j): [] for i in range(F) for j in range(C)}
    t = 0

    tar = Ventana(fig)
    parar = False
    for ep in range(24):
        if parar:
            break
        for di in range(N_DAT):
            if parar:
                break
            x = DATOS[di]
            eta = max(ETA_MIN, ETA0 * (1 - t / T))
            sig = max(SIG_MIN, SIG0 * (1 - t / T))
            prev = W.copy()
            D = np.linalg.norm(W - x, axis=-1)
            bmu = tuple(np.unravel_index(np.argmin(D), D.shape))
            dr = np.linalg.norm(POS - np.array(bmu), axis=-1)
            hs = np.exp(-(dr ** 2) / (2 * sig ** 2))
            W = W + eta * hs[..., None] * (x - W)
            miembros[bmu].append((di, GRUPO[di]))
            mapa.actualizar(W, bmu=bmu, hs=hs, miembros=miembros)
            esp.actualizar(W, prev=prev, bmu=bmu, dato=x)
            bars.actualizar(eta, sig, t, T)
            estado.set_text(f"ÉPOCA {ep+1}/24  ·  dato {NOMB[di]}  ·  "
                            f"η = {eta:.3f}  ·  σ = {sig:.3f}")
            estado.set_color(CORAL if ep >= 10 else CREMA)
            t += 1
            if GUI_OK and not tar.refrescar(.004):
                parar = True
                break

    # resultado
    ok = tot = 0
    for b, ms in miembros.items():
        if not ms:
            continue
        dom = max(set(g for _, g in ms),
                  key=lambda g: sum(1 for _, gg in ms if gg == g))
        for _, g in ms:
            tot += 1
            ok += (g == dom)
    pureza = ok / max(tot, 1)
    mapa.actualizar(W, miembros=miembros)
    esp.actualizar(W)
    estado.set_text(f"RESULTADO · pureza {pureza:.0%} · los dos grupos "
                    "quedaron en casillas separadas ✓")
    estado.set_color(VERDE)
    tarjeta(ax0, .10, .045, .80, .06, VERDE)
    ax0.text(.50, .075, "SIN etiquetas: la ESTRUCTURA emergió de los datos "
             "— eso es auto-organización",
             color=CREMA, fontsize=12.5, ha="center", va="center",
             fontweight="bold")
    print(f"    pureza tras 24 épocas = {pureza:.0%}")
    while not tar.cerrada:
        tar.refrescar(.03)
    cerrar(tar)
    print("    Fin de la convergencia.")


# ════════════════════════════════════════════════════════════════════════════
#  FASE 3 — CHULETA: las 5 fórmulas del SOM en una lámina
# ════════════════════════════════════════════════════════════════════════════
def fase_chuleta():
    aviso("CHULETA — las fórmulas del SOM en una lámina")
    fig, ax0 = figura(12.8, 7.4)
    titulo_ventana(fig, "SOM · Chuleta de fórmulas")
    ax0.text(.05, .94, "LAS FÓRMULAS DEL ALGORITMO DE KOHONEN",
             fontsize=22, color=CREMA, fontweight="bold", va="top")
    ax0.text(.05, .895, "lo que el ejercicio aplica con números, aquí de un "
             "vistazo", fontsize=11.5, color=DORADO, va="top")

    formulas = [
        ("PASO 1 · ENTRADA", TURQUESA,
         "x  =  ( x₁, x₂, …, xₙ )  normalizado a [0,1]",
         "cada dato llega como un vector; la normalización pone "
         "todos los rasgos en la misma escala."),
        ("PASO 2 · COMPETENCIA (BMU)", CORAL,
         "d(x, W) = √( Σₖ (xₖ − Wₖ)² )      BMU = argmin d",
         "todas las neuronas miden su distancia euclídea al dato; "
         "gana la Best Matching Unit."),
        ("PASO 3 · COOPERACIÓN", LILA,
         "h(r, r_BMU) = exp( − ‖r − r_BMU‖² / (2σ²) )",
         "la función topológica gaussiana: cuánto coopera cada "
         "neurona según su distancia EN LA MALLA a la BMU."),
        ("PASO 4 · ADAPTACIÓN", DORADO,
         "W(nuevo) = W + η · h · (x − W)",
         "regla de Kohonen: la BMU y su vecindario se acercan al "
         "dato, cada una en proporción a su h."),
        ("PASO 5 · CONVERGENCIA", VERDE,
         "η(t) = η₀·(1 − t/T)      σ(t) = σ₀·(1 − t/T)",
         "tasa de aprendizaje y radio decaen: primero aprende "
         "grueso (orden global), luego afina (detalle)."),
    ]
    y = .82
    for titulo, c, formula, nota in formulas:
        tarjeta(ax0, .045, y - .115, .91, .125, c)
        ax0.text(.07, y - .005, titulo, color=c, fontsize=12,
                 fontweight="bold", va="center")
        ax0.text(.34, y - .005, formula, color=CREMA, fontsize=12.5,
                 va="center", family="DejaVu Sans")
        ax0.text(.07, y - .082, nota, color=GRIS, fontsize=9.5, va="center")
        y -= .145

    tarjeta(ax0, .045, .045, .91, .075, GRIS)
    ax0.text(.50, .083, "se repite para cada dato × cada época → "
             "el mapa se ORDENA SOLO",
             color=CREMA, fontsize=11.5, ha="center", va="center",
             fontweight="bold")

    tar = Ventana(fig)
    while not tar.cerrada:
        tar.refrescar(.03)
    cerrar(tar)
    print("    Fin de la chuleta.")


# ════════════════════════════════════════════════════════════════════════════
#  MENÚ + MAIN
# ════════════════════════════════════════════════════════════════════════════
def men():
    fig, ax0 = figura(12.8, 7.2)
    titulo_ventana(fig, "SOM · Ejercicio numérico — Menú")
    ax0.text(.5, .93, "E J E R C I C I O   N U M É R I C O   D E L   S O M",
             fontsize=23, color=CREMA, ha="center", fontweight="bold",
             va="top")
    ax0.text(.5, .875, "todas las fórmulas de Kohonen, aplicadas con números "
             "y paso a paso", fontsize=12, color=DORADO, ha="center")
    ax0.text(.5, .83, "4 datos · 2 dimensiones · mapa 2×2 · 4 épocas",
             fontsize=11, color=GRIS, ha="center")

    est = {"accion": None}
    items = [
        ("1 · EJERCICIO PASO A PASO", fase_ejercicio, TURQUESA,
         "cada fórmula con sus números · pulsa ▸"),
        ("2 · CONVERGENCIA (24 épocas)", fase_convergencia, CORAL,
         "el mismo ejercicio, animado a fondo"),
        ("3 · CHULETA DE FÓRMULAS", fase_chuleta, DORADO,
         "las 5 fórmulas en una lámina"),
    ]
    y = .62
    for lbl, fn, c, sub in items:
        bax = fig.add_axes([.22, y, .56, .115])
        bax.set_facecolor(AZUL_2)
        bb = Button(bax, lbl, color=AZUL_2, hovercolor="#31497c")
        bb.label.set_color(CREMA); bb.label.set_fontsize(13)
        for sp in bax.spines.values():
            sp.set_color(c); sp.set_linewidth(2.4)
        bax.zorder = 40
        bb.on_clicked(lambda e, f=fn: est.__setitem__("accion", f))
        ax0.text(.5, y - .045, sub, color=GRIS, fontsize=9.5, ha="center")
        y -= .185

    ax0.text(.5, .065, "cierra esta ventana para salir · la demo visual "
             "anterior sigue en demostracion_som.py",
             color=GRIS, fontsize=9, ha="center")

    tar = Ventana(fig)
    while not tar.cerrada:
        accion = est["accion"]
        est["accion"] = None
        if accion is None:
            tar.refrescar(.02)
            continue
        try:
            accion()
        except Exception as exc:
            print(f"    \033[31m(sección interrumpida: {exc})\033[0m")
    cerrar(tar)
    print("\n  ¡Hasta la próxima! — que los números ordenen solos.")


def principal():
    print("\n\033[1;36m" + "=" * 74)
    print("  EJERCICIO NUMÉRICO · MAPAS AUTO-ORGANIZADOS DE KOHONEN (SOM)")
    print("=" * 74 + "\033[0m")
    print(f"  backend gráfico: {BACKEND}")
    print("  ▸ Se abrirá el MENÚ: elige el ejercicio paso a paso, la")
    print("    convergencia animada o la chuleta de fórmulas.")
    if not GUI_OK:
        print("  \033[33m⚠ sin ventana gráfica: el ejercicio se resuelve e")
        print("    imprime completo en la terminal.\033[0m")
    print()
    try:
        men()
    except KeyboardInterrupt:
        print("\n    ¡Ejercicio detenido por el usuario! Hasta pronto 👋")
    plt.close("all")
    print("\n\033[1;36m  ═══ EJERCICIO TERMINADO — el mapa ordena solo ═══\033[0m\n")


if __name__ == "__main__":
    principal()
