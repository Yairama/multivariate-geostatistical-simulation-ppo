
# MineRL-NPV: RL para planificar minado maximizando NPV

> Entrena una IA que decide qué bloque minar en cada paso para **maximizar el NPV** bajo incertidumbre geológica, con restricciones geométricas/operativas y visualizador 3D del block model. Loguea todo en **TensorBoard**.
> Basado en la idea del paper de Avalos & Ortiz (2023) que integra **simulación geoestadística multivariada + Deep RL** para scheduling a cielo abierto.

## TL;DR (stack)

* **Python 3.11+**
* **Gymnasium** (ambiente de RL)
* **Stable-Baselines3** + **sb3-contrib (MaskablePPO)**
* **PyTorch** (policy + extractor 3D CNN)
* **PyVista** o **vedo** (VTK) para **visualización 3D voxel**
* **TensorBoard** para métricas/curvas/videos

---

## Por qué **MaskablePPO** (y no DQN)

**Problema:** acción discreta masiva = elegir una “columna” (x,y) superficial válida entre `Nx*Ny` posiciones, pero **con máscara dinámica** (solo acciones factibles por precedencias de pit). Observación = **volumen 3D** con muchos canales (leyes medias/std, flags de UPL, revenue factor, estados dinámicos), justo como el paper.

* **DQN** (Q-learning) funciona con acciones discretas, pero:

  * No trae **action masking** nativo en SB3 (toca hacks/penalizaciones).
  * Se pone inestable con **espacios de acción grandes** y máscara cambiante.
* **PPO** on-policy es estable y, con **`MaskablePPO` (sb3-contrib)**, soporta **máscara de acciones inválidas** out-of-the-box.
* En la práctica, **MaskablePPO** + **extractor 3D** es la ruta más directa y estable para este setup grande, en línea con el marco del paper (secuencia día a día, reward económico descontado al NPV, precedencias geométricas).

> Conclusión: **usamos MaskablePPO**. Si luego quieres comparar, el proyecto deja ganchos para probar DQN.

---

## Data requerida

Mínimo, por **block model** (grilla **Nx × Ny × Nz**; cada celda = bloque):

1. **Geología/geoestadística (Left State)**

   * Para cada bloque:

     * **Mean** y **Std** de **8 leyes** (ejemplo del paper: `Cu, Fe, S, C, Al, Na, As, K`).
   * (Opcional fuerte) **R realizaciones** multivariadas (para inyectar **incertidumbre** por episodio).
   * Soporta dataset real o **generador sintético** incluido.

2. **Proyecto/operaciones (Right State dinámico)**

   * **UPL flag** (1 si dentro del pit final, 0 si fuera).
   * **Revenue factor mínimo** al que el bloque paga (para restricciones económicas).
   * **Estado de extracción** (día de extracción, destino, etc. → se actualiza durante el episodio).
   * **Parámetros económicos y metalúrgicos** (puedes usar los del paper como default):

     * Capacidad mina: \~**425k t/día** (≈ 40 bloques de 16 m³ por día)
     * Recuperaciones (ejemplo del paper):

       * **Oxidos**: `RecOXCu = 93.4 + 0.7*(Cu/S) - 20*As` (tope 95%, piso 40%)
       * **Sulfuro**: `RecSULCu = 80.0 + 5*(Cu/S) - 10*As` (tope 95%, piso 50%)
     * **GAC** (ácido para óxidos): `GAC = 25.4 + 18.8*C` USD/t
     * **Precio Cu**: `2.3 USD/lb`, **Costo minado**: `6 USD/t`, **Chancado+molienda**: `15 USD/t`, **Proc.**: `0.5 USD/lb`, **descuento**: `15% anual`

> El reward inmediato = **ingreso – costos** del bloque/día (sin descuento), y PPO aplica descuento temporal con `gamma = 1/(1+d)` para alinear con **NPV** como en el paper.

Formato soportado:

* **Parquet/NPZ** con arrays densos:

  * `left_mean[Nx,Ny,Nz,8]`, `left_std[Nx,Ny,Nz,8]`
  * `upl[Nx,Ny,Nz]`, `rev_factor[Nx,Ny,Nz]`
  * `realizations[R,Nx,Ny,Nz,8]` (opcional)
* O un **loader** para CSVs/GeoTIFFs/etc. (incluimos ejemplo).

---

## Arquitectura del proyecto

```
mine_rl_npv/
├─ data/
│  ├─ sample/            # dataset sintético listo para probar
│  └─ your_mine/         # tus datos reales
├─ configs/
│  ├─ env.yaml           # tamaños de grid, costos, recoveries, discount, etc.
│  └─ train_maskppo.yaml # hiperparámetros PPO, timesteps, logging
├─ envs/
│  ├─ mining_env.py      # Gymnasium Env + máscara de acciones
│  └─ geometry.py        # precedencias (cross-shape 5), pit angles, etc.
├─ rl/
│  ├─ feature_extractor_3d.py  # CNN 3D (PyTorch) para estados volumétricos
│  ├─ train.py           # loop de entrenamiento (SB3 + TensorBoard)
│  └─ evaluate.py        # eval sobre múltiples realizaciones (NPV dist)
├─ viz/
│  ├─ viewer.py          # visualizador 3D (PyVista/vedo)
│  └─ tb_video.py        # render de episodios a TensorBoard (gif/mp4)
├─ geo/
│  ├─ synth_generator.py # genera pórfido sintético (demo)
│  └─ loaders.py         # IO de datos (parquet/npz)
├─ experiments/
│  └─ runs/...           # logs y modelos
├─ notebooks/
│  └─ sanity_checks.ipynb
└─ README.md
```

---

## Instalación

```bash
# Recomendado: entorno virtual
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

pip install -U pip
pip install gymnasium stable-baselines3 sb3-contrib torch torchvision tensorboard
pip install pyvista vedo imageio[ffmpeg] numpy pandas pyyaml
# En Linux/macOS puede requerir system libs de VTK (PyVista/vedo). En Windows suele “just work”.
```

---

## Cómo funciona el ambiente (Gymnasium)

* **Observación** (tensor 4D): concat de **Left State** y **Right State** por canales, shape `(C, Nx, Ny, Nz)`:

  * Left: `mean(8) + std(8)` → 16 canales
  * Right din.: `extracted_flag`, `day_extracted`, `destino`, `feed_grades_hist` (8), `UPL`, `rev_factor` → aprox. 12–16 canales
* **Acción**: índice discreto en `[0, Nx*Ny)` que mapea a una **columna (x,y)**.

  * El bloque efectivo es el **bloque superficial** de esa columna (z top) **si** cumple **UPL + precedencias**.
* **Máscara de acciones**: solo habilita `(x,y)` **factibles** (implementamos algoritmo tipo **cross-shape (5)** + pendientes/bench) como en el paper.
* **Reward**: valor económico del bloque en ese paso (ingreso – costos). **Sin descuento** aquí; el descuento va en `gamma`.
* **Termina** cuando no quedan acciones factibles (mina agotada) o se cumple **capacidad diaria** por step → `step` avanza “día”.
* **Reset (nuevo episodio)**: se reinicia estado dinámico y opcionalmente se **samplea una realización** distinta como “ground truth” para esa corrida (o se usa **E-type**). El paper explora ambas opciones.

> En `sb3-contrib`, envolvemos con `ActionMasker` para exponer `action_mask` a **MaskablePPO**.

---

## Entrenamiento

```bash
# 1) Generar/colocar data
python -m geo.synth_generator --out data/sample   # o copia tus datos a data/your_mine

# 2) Entrenar
python -m rl.train --config configs/train_maskppo.yaml --env configs/env.yaml --data data/sample

# 3) TensorBoard
tensorboard --logdir experiments/runs
```

**Logs**: recompensa por episodio, **NPV estimado**, % de acciones inválidas evitadas, grados de feed, proporción de mineral/vacías, etc. También **videos** de episodios renderizados desde el viewer.

---

## Visualización 3D

```bash
# Reproducir un episodio con la política actual
python -m viz.viewer --model experiments/runs/<tu_run>/models/best.zip --env configs/env.yaml --data data/sample
```

* Render **voxel** del block model:

  * Colores por **estado**: no minado / minado / destino (óxido vs sulfuro) / grado Cu.
  * Reproduce **día a día** la secuencia (puedes pausar/adelantar).
* Opción de **exportar** MP4/GIF y logearlo a **TensorBoard** (`viz/tb_video.py`).

---

## Configuración de costos/recuperaciones (default del paper)

En `configs/env.yaml` vienen por defecto los parámetros de la sección de caso de estudio (editables).
Incluyen **capacidad mina**, **recuperaciones** por planta (fórmulas con límites), **GAC**, **precios** y **descuento** para alinear `gamma = 1/(1+d)` con NPV.

> Nota: el paper usa aprendizaje sobre secuencias diarias y **descuento económico** directamente; nosotros mantenemos esa lógica para que el retorno acumulado ≈ **NPV**.

---

## Roadmap (tareas)

### Núcleo RL

* [ ] `MiningEnv` (Gymnasium): reset/step, máscara de acciones, cálculo de reward, límite de capacidad por día.
* [ ] **Precedencias geométricas** (cross 5, bench height, slope one-down) y UPL (solo mina si `UPL=1`).
* [ ] **Destino de bloque** por mejor valor: óxidos vs sulfuros (aplica reglas de Cu/S, C y revenue).
* [ ] **Costo ácido GAC** en óxidos, costos de minado/proceso y precio Cu.
* [ ] **Descuento** con `gamma = 1/(1+d)` (e.g., `d=0.15 anual`, convertir a diario).
* [ ] **Sampler de realizaciones** vs **E-type** por episodio (como evalúa el paper).
* [ ] **FeaturesExtractor 3D** (PyTorch): CNN3D → MLP (custom policy SB3).
* [ ] Entrenamiento con **MaskablePPO** (sb3-contrib), `ActionMasker`.

### Visualización / Logging

* [ ] Viewer 3D (PyVista/vedo) con animación por día y paletas por estado/grade.
* [ ] Export de frames a **TensorBoard** (videos de episodios).
* [ ] Métricas: retorno, **NPV**, feed grade, % waste, cumplimiento de capacidad, hist de destinos, etc.

### Datos / IO

* [ ] Loader de **parquet/npz**.
* [ ] Generador sintético (porfírico): gradientes, domos, ruido correlacionado.
* [ ] (Opcional) Hook para cargar realizaciones de simulación multivariada existentes.

### Evaluación / Ciencia

* [ ] Script `evaluate.py`: corre N episodios sobre múltiples realizaciones y reporta distribución de **NPV** (como en el paper).
* [ ] Barridos de hiperparámetros (lr, clip PPO, `ent_coef`, tamaño de red 3D).
* [ ] Comparativa **E-type vs single-realization** al entrenar/evaluar (paper muestra diferencias y robustez).
* [ ] (Opcional) Implementar restricciones operativas adicionales (frentes, mínimo área/bench) sugeridas en la discusión del paper.

---

## Ejemplo de config (recorte)

```yaml
# configs/env.yaml
grid:
  nx: 64
  ny: 64
  nz: 32
physics:
  density_t_per_block: 10000        # ejemplo
mine:
  daily_capacity_blocks: 40          # ~425 kt/día si bloques ~16m
economics:
  price_cu_usd_per_lb: 2.3
  mining_cost_usd_per_t: 6.0
  proc_cost_usd_per_t: 15.0
  proc_cost_usd_per_lb: 0.5
  annual_discount_rate: 0.15
  steps_per_year: 365
met:
  rec_oxide: "clip(93.4 + 0.7*(cu/s) - 20*as, 40, 95)"
  rec_sulf:  "clip(80.0 + 5.0*(cu/s) - 10*as, 50, 95)"
  gac_usd_per_t: "25.4 + 18.8*c"
routing_rules:
  oxide_if:
    - "c <= 0.5"
    - "cu/s >= 0.4"
    - "rev_oxide > rev_sulf"
geometry:
  precedence: "cross5"  # vecinos cardinales
  bench_height_blocks: 1
  slope_down: 1
```

---

## Comandos útiles

```bash
# Entrenar (con logs a TensorBoard)
python -m rl.train --config configs/train_maskppo.yaml --env configs/env.yaml --data data/sample

# Evaluar NPV en 100 realizaciones (si existen) y guardar histograma
python -m rl.evaluate --model experiments/runs/<run>/models/best.zip --data data/sample --episodes 100

# Visualizar episodio
python -m viz.viewer --model experiments/runs/<run>/models/best.zip --env configs/env.yaml --data data/sample

# Abrir TensorBoard
tensorboard --logdir experiments/runs
```

---

## Notas de diseño/teoría (ligadas al paper)

* **Estado dividido** (Left vs Right) y **CNN 3D** para extraer patrones espaciales del block model (igual espíritu del paper).
* **Acción** = elegir bloque superficial factible; la máscara **evita acciones inválidas** (precedencias + UPL).
* **Reward** alinea retorno acumulado con **NPV** usando `gamma = 1/(1+d)` (valor del dinero en el tiempo).
* **Incertidumbre**: usar E-type o samplear realizaciones por episodio; el paper muestra que ambos caminos pueden aprender políticas competitivas y evalúa NPV sobre muchas realizaciones.
* **Restricciones extra** (operabilidad de palas, mínimo área por bench, múltiples frentes): están en la sección de discusión del paper y pueden añadirse en `geometry.py`.



Perfecto 🙌 ahora que me dices las **cabeceras reales de tu block model** (`x,y,z,ton,clays,chalcocite,bornite,chalcopyrite,tennantite,molibdenite,pyrite,cu,mo,as,rec,bwi`), te rearmo el `README.md` adaptado a esa data (en vez del ejemplo del paper con 8 leyes).

---

# MineRL-NPV: RL para planificar minado maximizando NPV

> Entrena una IA que decide qué bloque minar en cada paso para **maximizar el NPV**, con restricciones geométricas y metalúrgicas, usando **tu block model** real. Incluye un visualizador 3D voxel y loguea todo en **TensorBoard**.
> Basado en el marco de Avalos & Ortiz (2023).

---

## 📂 Datos de entrada

**block_model.csv** :

| Columna        | Descripción breve                                        |
| -------------- | -------------------------------------------------------- |
| `x, y, z`      | Coordenadas enteras del bloque (grid index)              |
| `ton`          | Toneladas del bloque                                     |
| `clays`        | % arcillas u otra variable geometalúrgica                |
| `chalcocite`   | % de calcosina                                           |
| `bornite`      | % de bornita                                             |
| `chalcopyrite` | % de calcopirita                                         |
| `tennantite`   | % de tenantita                                           |
| `molibdenite`  | % de molibdenita                                         |
| `pyrite`       | % de pirita                                              |
| `cu`           | % Cu total                                               |
| `mo`           | % Molibdeno                                              |
| `as`           | % Arsénico                                               |
| `rec`          | Recuperación metalúrgica estimada (si ya está calculada) |
| `bwi`          | Bond Work Index (indicador de energía de molienda)       |

👉 De estos, el agente usará:

* **Features geológicos/metalúrgicos** (`clays,...,pyrite,cu,mo,as,bwi`).
* **Variables de negocio** (`ton`, `rec` o fórmulas de recuperación si quieres redefinirlas).
* **Coordenadas `x,y,z`** solo para indexar la grilla 3D y aplicar precedencias geométricas.

---

## 🔍 Observación y Acción

* **Observación (estado del agente)**: tensor 3D `(C, Nx, Ny, Nz)` con canales:

  * Leyes y atributos: `clays, chalcocite, bornite, chalcopyrite, tennantite, molibdenite, pyrite, cu, mo, as, bwi` (11 canales).
  * Dinámicos: bloque minado sí/no, día de extracción, destino (planta ox vs sul), etc.
  * Recuperación (`rec`) puede entrar como feature inicial o calcularse on-the-fly con fórmulas.
* **Acción**: elegir columna `(x,y)` factible → se extrae el bloque superficial disponible.
* **Máscara de acciones**: asegura que solo se pueda minar bloques que cumplen **UPL + precedencias** (cross-shape 5).

---

## ⚖️ Recompensa

El **reward inmediato** = `Ingreso – Costos` del bloque/día:

* **Ingreso**: `(ton * cu% * rec * precio_Cu) + (ton * mo% * rec_mo * precio_Mo)`
* **Costos**: `ton * (costo_mina + costo_proc(bwi,clays) + costo_fijo)`

  * `costo_proc` puede escalar con `bwi` (índice de molienda) o penalización por `clays`.
* **Descuento**: PPO usa `gamma = 1/(1+d)` → se alinea con el NPV.

👉 Si no quieres usar `rec` directo de la columna, puedes entrenar reglas:

* Ejemplo: `rec_cu = f(cu, as, clays, mineralogía)`
* Ejemplo: `rec_mo = f(mo, mineralogía)`

---

## 🚀 Algoritmo de RL

* **MaskablePPO** (`sb3-contrib`) = elección principal.

  * Estable con espacios de acción grandes.
  * Soporta **masking dinámico** de acciones inválidas (bloques no factibles).
* **Extractor de features**: CNN3D que toma el volumen y saca embeddings espaciales.

---

## 📊 TensorBoard

Se logean:

* Reward acumulado (NPV estimado).
* Distribución de feed (`Cu`, `Mo`, `As`).
* % de waste vs mineral.
* Histograma de destinos.
* Videos voxel de episodios.

---

## 📦 Estructura del proyecto

```
mine_rl_npv/
├─ data/
│  ├─ sample_model.csv     # ejemplo con cabeceras dadas
├─ configs/
│  ├─ env.yaml             # parámetros económicos y de proceso
│  └─ train.yaml           # hiperparámetros RL
├─ envs/
│  └─ mining_env.py        # Gymnasium env con máscara y reward
├─ rl/
│  ├─ train.py             # loop de entrenamiento SB3
│  ├─ evaluate.py          # evalua NPV en varios episodios
│  └─ feature_extractor.py # CNN3D PyTorch
├─ viz/
│  ├─ viewer.py            # visualización 3D voxel (PyVista/vedo)
│  └─ tb_video.py          # export de videos a TensorBoard
└─ README.md
```

---

## ⚙️ Configuración económica (ejemplo en `configs/env.yaml`)

```yaml
economics:
  price_cu_usd_per_lb: 2.3
  price_mo_usd_per_lb: 10.0
  mining_cost_usd_per_t: 6.0
  proc_base_cost_usd_per_t: 15.0
  annual_discount_rate: 0.15
  steps_per_year: 365

processing:
  cost_multiplier_bwi: 0.1     # penaliza molienda según BWI
  penalty_clays: 0.05          # penalización $ por % de arcilla
```

---

## 🔧 Roadmap de tareas

* [ ] Parser del CSV/parquet de bloques con cabeceras reales.
* [ ] Construcción del `MiningEnv` en Gymnasium.
* [ ] Implementar **reward económico** con `ton, cu, mo, as, rec, bwi, clays`.
* [ ] Implementar **máscara de acciones** por precedencias y UPL.
* [ ] Configurar `MaskablePPO` con extractor 3D.
* [ ] Visualizador voxel en 3D mostrando bloques minados.
* [ ] Export a TensorBoard con métricas + videos.
* [ ] Scripts de entrenamiento y evaluación (NPV por realizaciones).

---

¿Quieres que te arme un **ejemplo mínimo de `env.yaml` + reward formula exacta** usando tus columnas (`ton, cu, mo, as, rec, bwi, clays`), para que ya quede lista la lógica de ingresos y costos?
