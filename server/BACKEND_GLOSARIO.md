# Likelihood Backend — Glosario y Lógica Explicada (en Español)

> Este documento explica **qué es** cada concepto técnico y **por qué** se usa en el backend de Likelihood.
> Es la contrapartida en español de `BACKEND_CONCEPTS.md`, enfocada en definiciones y razonamiento.

---

## Índice

1. [Términos Fundamentales](#1-términos-fundamentales)
2. [Conceptos de Arquitectura](#2-conceptos-de-arquitectura)
3. [Sistemas de Ejecución Aislada](#3-sistemas-de-ejecución-aislada)
4. [Gestión de Estado y Persistencia](#4-gestión-de-estado-y-persistencia)
5. [Seguridad y Autenticación](#5-seguridad-y-autenticación)
6. [Métricas y Monitoreo](#6-métricas-y-monitoreo)
7. [Entrenamiento de Modelos](#7-entrenamiento-de-modelos)
8. [Datos y Conjuntos de Datos](#8-datos-y-conjuntos-de-datos)
9. [Documentación Generada](#9-documentación-generada)
10. [Términos Específicos del Proyecto](#10-términos-específicos-del-proyecto)

---

## 1. Términos Fundamentales

### FastAPI

**¿Qué es?** FastAPI es un framework de Python para construir **APIs REST** (interfaces de programación). Permite definir rutas HTTP (como `GET /api/models` o `POST /api/serving/deploy`) y devuelve respuestas en formato JSON.

**¿Por qué se usa aquí?** FastAPI es rápido (usa `async`/`await` para no bloquear), valida automáticamente los datos de entrada con **Pydantic**, y genera documentación automática. El backend de Likelihood lo usa como **punto de entrada HTTP**: todas las peticiones del navegador pasan por FastAPI antes de llegar a la lógica de negocio.

**Relación con el código:** `server_fastapi.py` crea el `FastAPI()` y registra las rutas. `serving_api.py` y `training_api.py` son **routers** (sub-conjuntos de rutas) que FastAPI agrega a la aplicación principal.

---

### Pydantic

**¿Qué es?** Pydantic es una biblioteca de Python para **validar datos**. Se usa para definir "modelos de datos": clases que describen la forma exacta que debe tener un dict (por ejemplo, un body de petición).

**¿Cómo funciona?** Cuando una clase Pydantic se define con campos tipados (`model_id: str`, `parameters: dict`), Pydantic:
1. **Valida** que los datos recibidos coincidan con los tipos esperados.
2. **Convierte** datos de texto a tipos correctos (por ejemplo, `"42"` → `42`).
3. **Genera errores** claros si algo no coincide.

**Ejemplo del proyecto:**
```python
class PlaygroundRequest(BaseModel):
    model_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
```
Si el cliente envía `{"model_id": 123}`, Pydantic acepta (123 se convierte a string). Si envía un objeto que no es dict en `parameters`, Pydantic rechaza la petición con un 422.

---

### JSON Schema

**¿Qué es?** JSON Schema es un **estándar** para describir la estructura de un documento JSON. Es como un "molde" que dice: "este campo es string", "este campo es obligatorio", "este campo puede tener estos valores".

**¿Por qué se genera programáticamente en el proyecto?** El proyecto genera el JSON Schema **automáticamente** a partir de los parámetros reales del modelo. Si un modelo tiene un parámetro `n` de tipo `int` con default `2`, el schema dice `{ "type": "integer", "default": 2 }`. Si mañana alguien cambia el default a `3`, el schema se actualiza solo. **Nunca hay un documento escrito a mano que se desactualice.**

---

### CORS (Cross-Origin Resource Sharing)

**¿Qué es?** CORS es un mecanismo de **seguridad del navegador** que controla qué sitios web pueden hacer peticiones HTTP a tu servidor desde otro dominio.

**¿Por qué importa?** Si la frontend (por ejemplo, en `localhost:3000`) hace una petición a `localhost:8000/api/models`, el navegador bloquea esa petición **a menos** que el servidor responda con el header `Access-Control-Allow-Origin: http://localhost:3000`.

**En el proyecto:** `cors_origins` es una lista de dominios permitidos. Si está vacía (por defecto), el servidor no envía headers CORS y el navegador bloquea peticiones cross-origin. Si el frontend y backend están en el mismo dominio, no se necesita CORS.

---

### Middleware

**¿Qué es?** Un middleware es un **capa intermedia** entre la petición entrante y la respuesta saliente. Intercepts la petición, hace algo (autenticar, loggear, comprimir), y la pasa al siguiente handler.

**¿Por qué se usa?** En lugar de repetir la lógica de autenticación en cada endpoint, se crea un middleware que **todas las peticiones** pasan. Si la API requiere un token, el middleware lo verifica **una vez** antes de que el handler reciba la petición.

**Middlewares del proyecto:**
- **`PrometheusMiddleware`** — cuenta cada petición (contador de requests, latencia).
- **`RateLimiterMiddleware`** — limita la velocidad por IP.
- **`AuthMiddleware`** — verifica el token de API.
- **`GZipMiddleware`** — comprime respuestas grandes.
- **`CORSMiddleware`** — agrega los headers CORS.

---

### Gzip (Compresión)

**¿Qué es?** Gzip es un algoritmo de **compresión de datos**. Reduce el tamaño de un archivo (o respuesta HTTP) al eliminar redundancia.

**¿Por qué se usa en APIs?** Una respuesta JSON de 10 KB puede comprimirse a 3 KB. Esto ahorra ancho de banda y reduce el tiempo de descarga.

**En el proyecto:** `gzip_min_bytes` define el umbral. Si la respuesta es menor que ese valor (por ejemplo, 500 bytes), no se comprime (la compresión tiene un overhead y no vale la pena para respuestas pequeñas). Si es mayor, se comprime.

---

### Rate Limiting (Token Bucket)

**¿Qué es?** Un rate limiter es un mecanismo que **limita la cantidad de peticiones** que un cliente puede hacer en un intervalo de tiempo.

**¿Cómo funciona el Token Bucket?**
1. Cada IP tiene un "barril" que se llena con `rate` tokens por segundo.
2. Cada petición consume 1 token.
3. Si el barril está vacío → **429 Too Many Requests**.
4. El barril se refill gradualmente.
5. `burst` es la capacidad máxima del barril (permite ráfagas cortas sin rechazar).

**En el proyecto:** El middleware asigna un barril por IP. Si un cliente hace 100 peticiones en 1 segundo y el rate es 20/s, las primeras 20 pasan (si burst=20), las siguientes 80 reciben 429.

---

### Percentile (Percentil)

**¿Qué es?** Un percentil es un valor que indica el **porcentaje de datos** que están por debajo de él.

- **P50** (mediana): el 50% de las latencias están por debajo de este valor.
- **P95**: el 95% están por debajo. Útil para detectar "casos extremos".
- **P99**: el 99% están por debajo. Detecta outliers raros.

**Ejemplo práctico:** Si P95 de latencia es 200ms, significa que 95 de cada 100 peticiones tardan menos de 200ms. Pero la P99 podría ser 5000ms — hay una petición que tardó 5 segundos, y eso es un problema real aunque el promedio sea 50ms.

**En el proyecto:** `monitoring_core.py` calcula P50/P90/P95/P99 de latencia y compute time para cada deployment. Esto da una visión mucho más completa que el simple promedio.

---

### Atomic Operation (Operación Atómica)

**¿Qué es?** Una operación atómica es una que **no puede ser interrumpida a la mitad**. Es todo o nada: o se completa por completo, o deja el estado anterior intacto.

**¿Por qué es importante aquí?** Si el servidor escribe `deployments.json` y se apaga en medio de la escritura, el archivo queda corrupto (medio JSON). Con `os.replace()` (operación atómica en POSIX), el archivo es **completamente el nuevo** o **completamente el antiguo**. Nunca un híbrido.

**En el proyecto:** `_atomic_write()` escribe a un archivo temporal primero, luego `os.replace()` mueve el temporal sobre el definitivo. Si el proceso muere entre ambas operaciones, el archivo original sigue intacto.

---

### Pickle

**¿Qué es?** Pickle es el formato de **serialización binaria** nativo de Python. Convierte un objeto Python (una lista, un dict, una instancia de clase) en bytes y viceversa.

**¿Por qué es peligroso?** Un attacker puede crear un pickle malicioso que ejecute código al deserializar. Por eso, el proyecto **nunca** deserializa pickles de origen no confiable: siempre se ejecutan en un **sandbox** (subproceso aislado).

**En el proyecto:** El sandbox escribe el modelo entrenado como pickle al disco (`save_path`). El `loader` lo lee después para hacer predicciones. El proceso de lectura también ocurre **dentro del sandbox**.

---

### DataFrame (pandas)

**¿Qué es?** Un DataFrame es una **tabla bidimensional** (filas × columnas) con tipos de datos heterogéneos (números, texto, fechas). Es la estructura principal de la biblioteca `pandas`.

**¿Por qué se usa para datasets?** Los conjuntos de datos (CSV, Excel, Parquet) son tablas. Pandas los lee directamente a un DataFrame. Es mucho más eficiente que un dict anidado.

**En el proyecto:** `training_datasets.py` lee archivos CSV/XLSX/Parquet en un DataFrame. Se guarda **en memoria** (RAM) del servidor. El `jobs.json` solo guarda una referencia (ID + columnas seleccionadas), no el DataFrame completo. Esto mantiene el archivo JSON pequeño incluso con 20,000 filas.

---

### Docstring

**¿Qué es?** Una docstring es un **string** que se escribe al inicio de una función, clase o módulo para describirlo. Es el equivalente de un comentario que el sistema de documentación (`python -m pydoc`) extrae automáticamente.

**¿Por qué importa aquí?** El catálogo (AST) extrae las docstrings de las clases y métodos. El `description` de cada modelo en el catálogo viene directamente de la docstring. Si el desarrollador no escribe docstring, el `description` queda vacío y la UI no muestra nada.

---

### AST (Abstract Syntax Tree)

**¿Qué es?** Un AST es la **representación en árbol** de la estructura sintáctica de un programa. Cuando Python lee `def f(x: int) -> int: return x + 1`, el `ast.parse()` produce:

```
FunctionDef(name='f', args=[arg(x, annotation=Name(id='int')), ...], body=[Return(value=BinOp(...))])
```

**¿Por qué se usa en lugar de importar?** Si importas `likelihood.models.arima`, Python ejecuta **todo** el módulo: `import numpy`, `from likelihood import ...`, etc. Si no tienes numpy instalado, el import falla. Con `ast.parse()`, Python **solo lee el texto** del archivo. No ejecuta nada. No necesita numpy. Funciona en cualquier máquina.

**En el proyecto:** `model_catalog.py` usa `ast.parse()` para leer cada `.py` de `likelihood/models/`. Extrae nombres de clases, métodos, parámetros con tipos, defaults y docstrings. **Sin ejecutar ninguna línea de código.**

---

### Variadic Parameter (Parámetro Variádico)

**¿Qué es?** Un parámetro variádico es uno que acepta **un número variable** de argumentos. En Python, se declara con `*args` (posicional) o `**kwargs` (por nombre).

**Ejemplo:** `def predict(*args, **kwargs)` puede llamarse con 0, 1, 2... argumentos.

**¿Por qué se omiten en la UI?** Un parámetro variádico no se puede representar con un único widget (slider, checkbox, texto). No tiene un tipo fijo ni un default. Por eso `method_fields()` lo **skipea**: no aparece en el formulario de invocación.

---

### Literal

**¿Qué es?** Un literal es un **valor constante** escrito directamente en el código: `42`, `"hola"`, `True`, `[1, 2, 3]`, `{"a": 1}`.

**¿Por qué se valida?** En el generador de notebooks, los defaults de parámetros se insertan en el código generado. Si un default es `f"usuario_{os.getlogin()}"` (no es literal), ejecutar el notebook ejecutaría `os.getlogin()` en la máquina del usuario. El check `_is_safe_literal()` usa AST para verificar que el texto **solo** contiene literales puros.

---

### Widget

**¿Qué es?** En el contexto de la UI, un widget es un **componente interactivo** del formulario. Cada tipo de parámetro del modelo se mapea a un widget:

| Tipo de parámetro | Widget | Ejemplo |
|---|---|---|
| `int` con rango | **Slider** | `n` (orden) = 2 → slider de 1 a 10 |
| `float` | **Slider decimal** | `lr` = 0.01 → slider de 0 a 0.1 |
| `bool` | **Toggle** | `shuffle` = True → switch on/off |
| `str` | **Text input** | `name` = "mi_modelo" → campo de texto |
| `list/tuple` | **Deshabilitado** | `shape` = (128, 256) → no se puede editar en JSON |

**¿Por qué?** El formulario de invocación no puede pedirle al usuario que escriba `[128, 256]` en una casilla de texto. Es mejor deshabilitar ese campo y mostrarlo como valor fijo.

---

## 2. Conceptos de Arquitectura

### Single Responsibility Principle (Principio de Responsabilidad Única)

**¿Qué es?** Cada módulo/clase debe tener **una sola razón para cambiar**. Si un módulo hace tres cosas (guardar datos, validar, renderizar), debería dividirse en tres.

**Ejemplo en el proyecto:**
- `server_core.py` → "ejecutar modelos en sandbox" (NO guarda deployments).
- `serving_core.py` → "gestionar deployments" (NO ejecuta modelos en sandbox; **llama** a `server_core.run_invoke`).
- `monitoring_core.py` → "agregar métricas desde eventos" (NO graba eventos; **lee** los de `ServingStore`).

Si mañana necesitas cambiar cómo se ejecutan modelos, solo tocas `server_core.py`. No necesitas tocar `serving_core.py` ni `monitoring_core.py`.

---

### Domain Layer vs. Presentation Layer (Capa de Dominio vs. Capa de Presentación)

**¿Qué es?** La **capa de dominio** contiene la lógica de negocio (cómo entrenar, cómo desplegar, cómo ejecutar). La **capa de presentación** maneja lo que viene del navegador (HTTP requests, validación de body, auth).

**¿Por qué separar?**
- Puedes probar la lógica de dominio **sin** un servidor HTTP corriendo.
- Si mañana migras de FastAPI a Flask, solo cambias la capa de presentación.
- Los tests de la capa de dominio son más rápidos (no levantan un servidor).

**En el proyecto:** `training_core.py`, `serving_core.py`, `monitoring_core.py` **no importan FastAPI**. Solo `server_fastapi.py`, `serving_api.py`, `training_api.py` importan FastAPI.

---

### Dependency Injection (Inyección de Dependencias)

**¿Qué es?** En lugar de que una función **cree** sus dependencias (`store = TrainingStore()`), las **recibe** por parámetro (`def process(store: TrainingStore)`).

**¿Por qué importa?** En tests, puedes inyectar un `TrainingStore` falso (sin disco, sin subprocess) y verificar la lógica sin efectos secundarios. En producción, inyectas el store real.

**En el proyecto:** `TrainingStore` se crea una vez en `server_fastapi.py` (o en tests) y se **pasa** a `training_core.py`. `training_core.py` nunca hace `import training_datasets; store = DatasetStore()` directamente.

---

### In-Memory Cache (Caché en Memoria)

**¿Qué es?** Una copia de datos que se guarda en la **RAM** (memoria volátil) para evitar leer del disco en cada acceso.

**¿Por qué?** Leer `catalog.json` del disco en cada request sería lento. Si lo lees **una vez** al arrancar y lo guardas en RAM, cada acceso posterior es instantáneo (búsqueda en un dict Python).

**El "rehydrate":** Al arrancar el servidor, se lee el JSON del disco → se reconstruye el dict en RAM. Mientras el servidor está encendido, **no se vuelve a leer** del disco (excepto si se llama `refresh()`).

**En el proyecto:** `CatalogStore._catalog`, `ServingStore._models`, `TrainingStore._jobs` son los tres cachés en memoria.

---

### Dirty Flag (Bandera Sucia)

**¿Qué es?** Un booleano que marca si los datos han sido **modificados** desde la última vez que se guardaron al disco.

**¿Por qué?** No tiene sentido escribir `catalog.json` al disco cada vez que se modifica un campo. Con un dirty flag, solo se escribe cuando de verdad hay cambios.

**En el proyecto:** `CatalogStore.changed()` devuelve `True` si `_dirty` es `True`. El sistema de persistencia periodically llama `changed()` y solo persiste si es `True`.

---

### Reentrant Lock (Lock Reentrante)

**¿Qué es?** Un lock normal te impide entrar si **ya** tienes el lock. Un lock reentrante te permite **volver a pedir el lock** si ya lo tienes (mismo hilo).

**¿Por qué es importante aquí?** `deploy()` toma el lock → llama a `_persist_locked()` que **también** toma el lock → llama a `_record_error()` que **también** toma el lock. Si el lock no fuera reentrante, `deploy()` se bloquearía a sí mismo (**deadlock**). Con `RLock`, puede pedir el lock tres veces en el mismo hilo.

**En el proyecto:** `ServingStore` usa `threading.RLock()`.

---

### Bounded Semaphore (Semáforo Limitado)

**¿Qué es?** Un semáforo es un contador que permite **N** hilos/peticiones entrar simultáneamente. Si hay N+1 peticiones, la N+1 espera.

**¿Por qué se usa en deployments?** Si dos peticiones `deploy` intentan desplegar el **mismo** endpoint simultáneamente, una corruptaría la otra. Con un `BoundedSemaphore` por endpoint, solo una puede ejecutar el deploy a la vez. La otra espera.

**En el proyecto:** `self._semaphores[endpoint_id] = BoundedSemaphore(1)`. Solo 1 load/deploy por endpoint a la vez. Pero hay un `ThreadPoolExecutor` global que gestiona **varios** endpoints en paralelo.

---

### ThreadPoolExecutor (Pool de Hilos)

**¿Qué es?** Un pool de hilos es un conjunto de **N hilos** que ejecutan funciones en paralelo. En lugar de crear un hilo por tarea (que es caro), se reutilizan los N hilos del pool.

**¿Por qué se usa?** `ServingStore` puede tener 10 deployments activos. Cada petición predict puede ejecutar en paralelo usando los hilos del pool. `TrainingStore` puede tener 5 jobs de entrenamiento corriendo a la vez.

**En el proyecto:**
- `ServingStore` → `ThreadPoolExecutor(DEPLOY_WORKERS)` (por defecto 4).
- `TrainingStore` → `ThreadPoolExecutor(TRAIN_WORKERS)` (por defecto 2).
- `JobQueue` → `ThreadPoolExecutor(max_workers)` (lazy init).

---

### One-Shot vs. Persistent Session

**One-shot (una sola vez):**
- Se crea un subproceso → se ejecuta → se mata.
- El modelo se construye **de cero** en cada petición.
- Lento pero simple: no hay estado que gestionar.

**Persistent session (sesión persistente):**
- Se crea **un** subproceso que **sigue vivo**.
- El modelo se construye **una sola vez**.
- Cada petición es un JSON en stdin → respuesta en stdout.
- Rápido (no se reconstruye) pero complejo: hay que gestionar el ciclo de vida (¿murió el proceso? ¿timeout?).

**En el proyecto:** `run_invoke()` es one-shot. `ModelSession` es persistente. El UI usa one-shot por defecto; si el usuario quiere entrenar y predecir en secuencia, usa una session.

---

### Bootstrap String

**¿Qué es?** Un **string de Python** que se **concatena** al final del prelude del sandbox. Contiene la lógica principal de lo que va a hacer el subproceso.

**¿Por qué un string y no una función?** El subproceso es **otro proceso**. No puede importar funciones del servidor (no comparten memoria). La única forma de transferirle código es escribiéndole un script en su stdin. El bootstrap es ese script.

**En el proyecto:**
- `_PLAYGROUND_BOOTSTRAP` → "construye el modelo y devuelve su descripción".
- `_INVOKE_BOOTSTRAP` → "construye el modelo, llama a este método, devuelve el resultado".
- `_SESSION_BOOTSTRAP` → "construye el modelo una vez, luego lee JSON de stdin y responde".

---

### Health Check

**¿Qué es?** Un endpoint HTTP (por ejemplo `GET /health`) que responde **rápidamente** con un 200 si el servidor está vivo. Los sistemas de orquestación (Kubernetes, Docker) lo usan para saber si el contenedor está funcionando.

**¿Por qué no se autentiica?** Si el health check requiere token, un misconfiguración de la auth haría que el sistema de orquestación piense que el servidor está caído. El health check debe ser **siempre accesible**.

**En el proyecto:** `GET /health` devuelve `{"status": "ok"}` sin autenticación. La `AuthMiddleware` lo excluye explícitamente.

---

### SSRF (Server-Side Request Forgery)

**¿Qué es?** Un ataque donde un atacante usa tu servidor para hacer **peticiones HTTP internas** que no deberían ser visibles. Ejemplo: tu API acepta `url: "http://169.254.169.254/latest/meta-data/"` y tu servidor le hace el GET. El atacante obtiene datos de la infraestructura AWS que nunca deberían estar expuestos.

**¿Cómo se previene?** Validando que la URL **no** apunta a IPs privadas/reservadas:
- `10.0.0.0/8` (RFC1918)
- `172.16.0.0/12`
- `192.168.0.0/16`
- `169.254.0.0/16` (link-local)
- `127.0.0.0/8` (loopback)

**En el proyecto:** `training_datasets.py` resuelve el hostname con `socket.getaddrinfo()` y verifica el IP contra estas redes antes de hacer el fetch.

---

## 3. Sistemas de Ejecución Aislada

### Sandbox (Sandbox de Subproceso)

**¿Qué es?** Un sandbox es un **entorno aislado** donde se ejecuta código no de confianza. En este proyecto, es un **subproceso Python** con:

1. **Límites de recursos** (`resource.setrlimit`): CPU, memoria, archivos.
2. **Bloqueo de imports** (`sys.meta_path` hook): no se puede importar `os`, `subprocess`, `socket`, etc.
3. **Comunicación por pipes**: stdin/stdout con JSON.

**¿Por qué un subproceso y no un `thread`?** Un thread comparte **todo** con el proceso principal (memoria, variables, sockets). Si el modelo hace `import os; os.system('rm -rf /')`, destruye el servidor. Un **subproceso** es un **nuevo PID** con su propia memoria. Si mata el servidor, es solo el subproceso que muere; el servidor principal sigue vivo.

**¿Qué son los RLIMIT?**
- `RLIMIT_CPU` — segundos de CPU. Si el modelo entra en un loop infinito, se mata al llegar al límite.
- `RLIMIT_AS` — memoria máxima. Si el modelo intenta allocar 100 GB, el OS le da `MemoryError`.
- `RLIMIT_FSIZE` — tamaño máximo de archivo. Si intenta escribir un archivo de 10 GB, falla.
- `RLIMIT_NOFILE` — nº máx. de archivos abiertos.
- `RLIMIT_DATA` — segmento de datos máximo.

---

### `sys.meta_path` y Import Hook

**¿Qué es?** `sys.meta_path` es una **lista de importers** que Python consulta cuando ejecuta `import X`. Si uno de los importers devuelve un módulo, se usa. Si no, Python busca en `sys.path`.

**¿Cómo se bloquea un import?** Se instala un "finder" en `sys.meta_path[0]` que:
1. Recibe el nombre del módulo a importar.
2. Si está en la lista de bloqueados → `raise ImportError("import bloqueado")`.
3. Si no → `return None` (deja que Python siga buscando).

**En el proyecto:** `guarded_import` es ese finder. Bloquea `os`, `subprocess`, `socket`, `multiprocessing`, `threading`, `signal`, `ctypes`, `importlib`, etc.

---

### `hmac.compare_digest` (Comparación en Tiempo Constante)

**¿Qué es?** Comparar dos strings con `==` puede **filtrar información por tiempo**. Si el token esperado es `"abc123"` y el atacante prueba `"abc"`, Python compara `a==a` (1ms), `b==b` (1ms), `c==c` (1ms), `1!=1` (falla). El tiempo le dice al atacante que los primeros 3 caracteres son correctos.

**`hmac.compare_digest`** compara **byte a byte** en **tiempo constante** sin importar dónde difieren. Un atacante no puede deducir la posición del error.

**En el proyecto:** La auth middleware usa `hmac.compare_digest(provided_token, settings.api_token)`. Nunca `provided_token == settings.api_token`.

---

### Bootstrap y Prelude

**Prelude** (`_SANDBOX_PRELUDE`):
- Se ejecuta **primero**.
- Define las herramientas: `_emit()`, `_limit()`, `guarded_import`, `_convert_all()`.
- Configura los límites de recursos.
- Instala el import hook.

**Bootstrap** (`_INVOKE_BOOTSTRAP`, etc.):
- Se ejecuta **después** del prelude.
- Contiene la **lógica principal**: construir modelo, llamar método, devolver resultado.
- Se genera dinámicamente (incluye el nombre del modelo, el método, los argumentos).

**¿Por qué separar?** El prelude es **común** a todas las operaciones. El bootstrap es **específico** de cada operación. Separarlos permite mantener el prelude estable y generar bootstraps dinámicos sin duplicar el prelude.

---

## 4. Gestión de Estado y Persistencia

### Persistencia vs. Volatilidad

**Persistente (disco):** Datos que sobreviven al reinicio del servidor. `catalog.json`, `deployments.json`, `events.jsonl`, `jobs.json`.

**Volátil (RAM):** Datos que se pierden al apagar el servidor. El `DataFrame` en `DatasetStore`, los `ModelSession` activos, los jobs en `JobQueue`.

**¿Por qué esta separación?**
- Lo **crítico** (modelos desplegados, jobs de entrenamiento) debe sobrevivir.
- Lo **efímero** (dataframes, sesiones) no hace falta que sobreviva (se puede reconstruir).
- Guardar un DataFrame de 20,000 filas en JSON sería enorme y lento.

**En el proyecto:**
- `deployments.json` → persistente.
- `events.jsonl` → persistente.
- `jobs.json` → persistente (pero los jobs `running` se marcan `failed` al arrancar).
- `datasets` → **volátil** (el DataFrame está en RAM; el `jobs.json` solo guarda la referencia).
- `ModelSession` → **volátil** (el subproceso muere con el servidor).

---

### Append-Only Log (JSONL)

**¿Qué es?** Un archivo donde cada **línea** es un objeto JSON completo. Se **agrega** (append) al final; nunca se modifica una línea existente.

**¿Por qué JSONL y no JSON?** Un JSON es un **único** objeto. Si dos hilos quieren escribir simultáneamente, se corruptan. Con JSONL, cada hilo hace `f.write(json.dumps(event) + "\n")`. Las líneas no se superponen.

**En el proyecto:** `events.jsonl` es el log de monitoreo. Cada petición a un endpoint serving genera una línea. `EventLog.record()` hace el append. `_rewrite_locked()` compacta periódicamente (elimina eventos viejos, reescribe).

---

### Schema Migration

**¿Qué es?** Un **cambio en el formato** de los datos persistidos. Ejemplo: en la v1, los eventos tenían `model_id`. En la v2, se agrega `model_instance_id` y `catalog_model_id`.

**¿Cómo se migra?** Al arrancar, `_migrate_event_log()` lee todas las líneas. Si una línea tiene `model_id` pero no `model_instance_id`, se agrega `model_instance_id = model_id` y `catalog_model_id = model_id`. Se reescribe el archivo.

**¿Por qué no se rompe la compatibilidad?** Los eventos antiguos siguen siendo válidos (los campos nuevos son opcionales). Los clientes que no conocen `model_instance_id` siguen usando `model_id`.

---

### Artifact (Artefacto)

**¿Qué es?** Un **conjunto de archivos** que representa un modelo entrenado o un estado específico. Incluye:
- El archivo binario (HDF5, pickle, etc.).
- `artifact.json` (metadatos: modelo, método, parámetros, timestamp).
- Posiblemente archivos auxiliares (weights, optimizer state).

**¿Por qué se llama "artifact"?** En CI/CD, un "artifact" es el **resultado** de un proceso (build). Aquí, el artifact es el resultado del **entrenamiento**: el modelo + sus metadatos.

**En el proyecto:**
- `STORE_DIR/artifacts/<artifact_id>/` es la carpeta.
- El sandbox escribe el modelo binario.
- `training_core.py` agrega `artifact.json`.
- `serving_core.py` usa `artifact.json` para saber cómo cargar el modelo.

---

### Atomic Swap (Intercambio Atómico)

**¿Qué es?** Reemplazar un archivo/carpeta **completamente** sin un momento intermedio donde esté corrupto.

**Problema sin atomicidad:**
1. Escribes `artifact.json` (50% escrito) → servidor se apaga.
2. Al arrancar, `artifact.json` tiene medio JSON → parse falla → deployment no funciona.

**Solución:**
1. Escribes a `artifact.json.tmp` (completo).
2. `os.fsync()` (fuerza la escritura a disco).
3. `os.replace("artifact.json.tmp", "artifact.json")` → **atómico**.
4. Borras el `.tmp`.

**En el proyecto:** `_atomic_write()` y el swap de carpetas de artifacts usan esta técnica.

---

## 5. Seguridad y Autenticación

### API Token (Bearer Token)

**¿Qué es?** Un **secreto** que el cliente debe enviar en el header `Authorization: Bearer <token>` para autenticarse.

**¿Por qué se usa?** El proyecto es una API de modelos de ML. No se quiere que cualquier persona en internet pueda:
- Ejecutar métodos arbitrarios del modelo.
- Desplegar endpoints.
- Ver datos de entrenamiento.

**En el proyecto:** `LM_API_TOKEN` env var. Si está configurado, el `AuthMiddleware` verifica cada request. Si no está configurado, no hay auth (modo desarrollo).

### Timing Attack (Ataque por Tiempo)

**¿Qué es?** Un ataque que usa las **diferencias de tiempo** de ejecución para deducir información secreta.

**Ejemplo:**
```python
if user_input == secret:  # ❌ Comparación con ==
```
Si `user_input` y `secret` difieren en la posición 5, Python sale del `==` en 5 bytes. Si difieren en la posición 50, sale en 50 bytes. El atacante mide el tiempo y deduce que los primeros 5 caracteres son correctos.

**Solución:** `hmac.compare_digest(user_input, secret)` — siempre compara **todos** los bytes en tiempo constante.

---

### Constant-Time Comparison

Ver arriba. Siempre se usa `hmac.compare_digest` en lugar de `==` para comparar tokens.

---

### Import Blocking (Bloqueo de Imports)

**¿Qué es?** Impedir que un proceso ejecute `import os`, `import subprocess`, etc.

**¿Por qué es importante?** El sandbox ejecuta código de modelos que puede ser **malicioso** (o simplemente defectuoso). Si un modelo hace `import subprocess; subprocess.call("rm -rf /")`, el sandbox no debería permitirlo.

**En el proyecto:** `guarded_import` en `sys.meta_path` bloquea: `os`, `sys`, `subprocess`, `socket`, `multiprocessing`, `threading`, `signal`, `ctypes`, `importlib`, `importlib.machinery`, `pkgutil`, `pathlib`.

**Excepción:** El sandbox **sí** permite importar `numpy`, `pandas`, `tensorflow`, `torch`, etc. (las librerías ML que el modelo necesita). Solo bloquea las que dan acceso al sistema.

---

## 6. Métricas y Monitoreo

### Prometheus

**¿Qué es?** Prometheus es un **sistema de monitoreo** open-source (nace en 2016, antes en Google como "promtool"). Funciona así:

1. El servidor expone un endpoint `GET /metrics` con formato **Prometheus exposition format**:
   ```
   http_requests_total{method="GET",status="200"} 42
   http_request_duration_seconds{endpoint="invoke"} 0.15
   ```
2. Prometheus **scrapea** (hace GET a) ese endpoint cada N segundos.
3. Almacena las series temporales en su base de datos.
4. Grafana (u otra UI) grafica las series.

**¿Por qué se expone `/metrics` aquí?** Aunque el proyecto no corre Prometheus, el endpoint `/metrics` sigue el formato estándar. Esto permite:
- Conectar Prometheus después sin cambios.
- Hacer scripts de scrapeo.
- Exportar a otras herramientas (Grafana, Datadog, etc.).

**En el proyecto:** `GET /metrics` devuelve los contadores en formato Prometheus. `PrometheusMiddleware` incrementa los contadores en cada request.

### Counters vs. Gauges (Contadores vs. Medidores)

- **Counter:** Un número que **solo sube** (o se resetea). Ejemplo: `requests_total` → 1, 2, 3, ... nunca baja.
- **Gauge:** Un número que **sube y baja**. Ejemplo: `active_connections` → 5, 3, 7, 2, ...

**En el proyecto:**
- `requests_total`, `success_total`, `error_total` → **counters**.
- `latency_sum_ms` → **counter** (suma acumulada).
- `p95_latency_ms` → **gauge** (el percentil actual puede subir y bajar).

---

### Event-Driven Architecture (Arquitectura Basada en Eventos)

**¿Qué es?** En lugar de que el sistema "calcule" métricas activamente (polling), **los eventos** (peticiones que llegan) **disparan** la grabación de datos.

**Flujo:**
1. Petición llega → `ServingStore` graba un evento en `events.jsonl`.
2. `monitoring_core.py` **lee** los eventos y **calcula** métricas cuando se le pide (lazy).
3. No hay un demonio que "muestre" el servidor cada segundo.

**Ventaja:** Si no hay peticiones, no hay eventos, no hay cómputo. Los recursos se usan **solo** cuando hay actividad.

**En el proyecto:** `monitoring_core.aggregate()` se llama **on-demand** (cuando el endpoint `/api/serving/monitoring/overview` se pide). No hay un loop de fondo.

---

### Time Range Presets

**¿Qué es?** Presets de rango temporal predefinidos:
- `"1h"` → últimas hora.
- `"24h"` → últimas 24 horas.
- `"7d"` → últimos 7 días.
- `"30d"` → últimos 30 días.

**¿Por qué presets y no un input de fecha?** Un preset es más simple para el UI (un dropdown). Un input de fecha requiere validar formatos, fusos horarios, etc. Para monitoreo de una API, "última hora" es 90% de los casos.

---

### Bucketization (Bucketizar)

**¿Qué es?** Dividir un rango temporal en **intervalos** (buckets) y agregar métricas por intervalo.

**Ejemplo:** 24 horas con bucket de 15 min → 96 buckets. Cada bucket tiene `{requests, avg_latency, error_rate}`.

**¿Por qué?** Para graficar una **línea de tiempo**. Sin bucketización, tendrías un punto por petición (ruido). Con bucketización, tienes 96 puntos suaves que muestran la tendencia.

**En el proyecto:** `bucketize()` en `monitoring_core.py` divide eventos por `bucket_seconds`. Los buckets vacíos se rellenan con ceros (para que la gráfica no tenga huecos).

---

### Cohen's Kappa

**¿Qué es?** Cohen's kappa es una **métrica de acuerdo** entre dos clasificadores. Va de -1 a 1:
- `1.0` = acuerdo perfecto.
- `0.0` = acuerdo aleatorio (por azar).
- `-1.0` = acuerdo anti-perfecto (siempre se equivocan de forma sistemática).

**Fórmula:**
```
kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
```
Donde:
- `observed_agreement` = proporción de predicciones correctas.
- `expected_agreement` = proporción que se esperaría **por azar** (basada en la distribución de clases).

**¿Por qué no basta con accuracy?** Si el 95% de tus datos son clase A, un clasificador que siempre predice A tiene 95% accuracy pero kappa ≈ 0. Cohen's kappa **desconta** el acuerdo por azar.

**En el proyecto:** `training_metrics.py` implementa Cohen's kappa en Python puro (sin TensorFlow) para tests unitarios.

---

### Confusion Matrix (Matriz de Confusión)

**¿Qué es?** Una tabla NxN que cuenta cuántas veces se predijo cada clase siendo cada clase real.

Ejemplo (2 clases):
```
           Predicha
           Positivo  Negativo
Real  Positivo    80       5
     Negativo      7      108
```
- Verdaderos positivos: 80.
- Falsos positivos: 5.
- Falsos negativos: 7.
- Verdaderos negativos: 108.

**¿Por qué importa?** La accuracy (95%) no dice **dónde** se equivocó. La matriz de confusión muestra si el modelo confunde más la clase A con la B o viceversa.

**En el proyecto:** `training_metrics.py` implementa la matriz de confusión como referencia pura.

---

### RMSE, MAE, MAPE

Métricas de error para predicción de valores continuos:

- **RMSE** (Root Mean Squared Error): `sqrt(mean((pred - actual)^2))`. Penaliza mucho los errores grandes (al elevar al cuadrado). Unidad: misma que la variable predicha.
- **MAE** (Mean Absolute Error): `mean(|pred - actual|)`. Más interpretable (misma unidad). No penaliza tanto los errores grandes.
- **MAPE** (Mean Absolute Percentage Error): `mean(|pred - actual| / actual)`. En porcentaje. **Problema:** si `actual = 0`, division por cero. Por eso el proyecto **omite** pares donde `actual = 0`.

**En el proyecto:** `_series_metrics()` calcula las tres, skipando pares no finitos y cero en MAPE.

---

## 7. Entrenamiento de Modelos

### Framework Fit Detection (Detección de Fit de Framework)

**El problema:** `model_catalog.py` solo ve métodos definidos **dentro** de `likelihood/`. Si `class MyModel(keras.Model)`, el método `fit` está en `keras`, no en `likelihood`. El catálogo **no lo ve**.

**La solución (sin modificar el package):** `supports_framework_fit()` detecta que el modelo hereda de `keras.Model` (o similar) y **fabrica** un método sintético `fit` solo para la capa de entrenamiento. El código real no se toca.

**¿Por qué no modificar el package?** Porque el package es la **fuente de verdad** de la API pública. Si se modifica, los users que importan `likelihood.models.my_model` verían un método `fit` que no existe realmente (es un adapter interno). Mejor no contaminar la API pública.

---

### Keras Compile Spec

**¿Qué es?** Cuando entrenas un modelo Keras, primero lo "compilas" con:
- `optimizer` → cómo actualizar pesos (SGD, Adam).
- `loss` → cómo medir el error (MSE, CategoricalCrossentropy).
- `metrics` → qué métricas reportar (accuracy).

**¿Por qué el servidor lo hace?** El usuario no siempre sabe qué loss/optimizer usar. El servidor puede aplicar un **default** sensible (Adam + MSE) o permitir que el usuario especifique.

**En el proyecto:** `compile` es un dict en el job spec: `{"optimizer": "adam", "loss": "mse", "metrics": ["mae"]}`. El sandbox `_apply_compile()` lo aplica al modelo antes de `fit`.

---

### Epoch y Batch Size

- **Epoch:** Una **paseada** completa por todos los datos de entrenamiento. Si hay 10,000 muestras y `batch_size=100`, un epoch tiene 100 iteraciones.
- **Batch Size:** Cuántas muestras por iteración. 100 → el modelo ve 100 filas, actualiza pesos, ve las siguientes 100, etc.

**¿Por qué batch size?** 1 batch (SGD) → ruido alto pero barato. 100 (mini-batch) → equilibrio. 10,000 (batch full) → estable pero caro.

**En el proyecto:** `batch_size` es un parámetro del fit. Default 32.

---

### Validation Split

**¿Qué es?** Reservar una **porción** de los datos (por ejemplo 20%) para **medir** el modelo después de cada epoch, **sin** entrenar con ellos.

**¿Por qué?** Si entrenas con todos los datos, no sabes si el modelo **memorizó** (overfit) o **aprendió** (generaliza). Con validation, puedes ver la curva de loss de validación: si sube, el modelo está overfitteando.

**En el proyecto:** `validation_ratio` en `SeriesAnalysisRequest` (default 0.2). Los últimos 20% de la serie se reservan para validación.

---

### Early Stopping

**¿Qué es?** Un **callback** que detiene el entrenamiento si la métrica de validación **deja de mejorar** durante N epochs.

**¿Por qué?** Si la loss de validación ha subido 10 epochs seguidos, el modelo está overfitteando. No hay sentido seguir entrenando (gasta tiempo sin mejorar).

**En el proyecto:** `_training_callbacks()` agrega `EarlyStopping` y `ReduceLROnPlateau` al job de entrenamiento.

---

### Overfitting

**¿Qué es?** Cuando un modelo **memoriza** los datos de entrenamiento en lugar de **aprender** patrones. Se le va bien en los datos de entrenamiento pero **mal** en datos nuevos.

**Síntoma:** Loss de entrenamiento baja mucho, loss de validación **sube** después de un punto.

**Solutions:** Validación cruzada, regularización, early stopping, más datos.

**En el proyecto:** La CV y el `validation_ratio` ayudan a detectar overfitting. El EarlyStopping lo previene.

---

## 8. Datos y Conjuntos de Datos

### Dataset (Conjunto de Datos)

**¿Qué es?** Un **conjunto de filas** (muestras) con **columnas** (variables/atributos). Cada fila es una instancia; cada columna es una característica.

Ejemplo:
```
fecha       temperatura  humedad  viento_km/h
2024-01-01    15.2       60       25
2024-01-02    14.8       55       30
...
```

**En el proyecto:** Un "dataset" es un DataFrame en RAM con una referencia en `jobs.json`. Se puede subir (CSV/XLSX/Parquet) o fetchear por URL.

---

### Target y Features (Objetivo y Características)

- **Target (Y):** La variable que quieres **predecir**. (ej: `temperature`).
- **Features (X):** Las variables que **predicen** el target. (ej: `humidity`, `wind_kph`).

**En el proyecto:** El usuario selecciona columnas como `target` y el resto como `features`. El entrenamiento usa `X` para predecir `Y`.

---

### Normalization (Escalado)

**¿Qué es?** Convertir los datos a una escala **estándar** para que el algoritmo de ML no se confunda.

- **Standard (z-score):** `(x - mean) / std`. Resultado: media 0, desviación 1.
- **Min-Max:** `(x - min) / (max - min)`. Resultado: entre 0 y 1.
- **None:** No se escala.

**¿Por qué importa?** Si `temperature` va de 0 a 40 y `humidity` va de 0 a 100, un algoritmo como Adam puede converger más lento si las escalas son diferentes. El escalado ayuda.

**En el proyecto:** `normalization` en el job spec. Se aplica **solo** al target (no a features, que ya están en su escala natural).

---

### Time Series (Serie Temporal)

**¿Qué es?** Datos **ordenados por tiempo**. Cada fila tiene un timestamp. El **orden importa** (la fila de enero no puede ir antes de la de diciembre).

**¿Por qué no se hace sampling aleatorio?** Si tomas 10 filas aleatorias de 365 días, puedes caer en 2 días de enero y 8 de diciembre. No representas la variación estacional. Por eso `_limit_series()` usa `tail()` (las últimas N filas) en lugar de `sample()`.

**En el proyecto:** `SeriesAnalysisRequest` usa `validation_ratio` (la parte **final** de la serie es la validación, no una muestra aleatoria).

---

## 9. Documentación Generada

### Jupyter Notebook

**¿Qué es?** Un **documento interactivo** de Python. Tiene **celdas** (bloques): celdas de código (se ejecutan), celdas de markdown (texto formateado con LaTeX, imágenes, tablas).

**¿Por qué generarlos programáticamente?** Si un notebook se escribe a mano, y el modelo cambia (nuevo parámetro, default distinto), el notebook queda obsoleto. Si se **genera** del AST del modelo, siempre está actualizado.

**En el proyecto:** `notebook_generator.py` lee el catálogo (AST) → genera celdas → valida con `nbformat` → escribe `.ipynb`.

### nbformat

**¿Qué es?** La **especificación oficial** (schema) de un notebook Jupyter. Define cómo debe verse un `.ipynb` como JSON:
```json
{
  "cells": [{"cell_type": "markdown", "source": ["# Hola\n"]}],
  "metadata": {"kernelspec": {"name": "python3"}},
  "nbformat": 4,
  "nbformat_minor": 5
}
```

**¿Por qué validar?** Si el generador tiene un bug (por ejemplo, un celda sin `source`), el notebook no se abrirá en Jupyter. `nbformat.validate()` detecta errores **antes** de escribir el archivo.

**En el proyecto:** Si `nbformat` está instalado, se valida. Si no, se usa un dict plano (el schema es conocido pero no se valida en runtime).

---

## 10. Términos Específicos del Proyecto

### `server_core.py` (Núcleo del Servidor)

**Rol:** El módulo **compartido** por todo el backend. Contiene:
- Sandbox y bootstrap strings.
- `run_invoke()` y `ModelSession`.
- Loader resolution.
- Metrics.
- HTML helpers.
- `CatalogStore`.

**Principio:** Si vas a poner algo en `server_core.py`, debe ser **usado por más de un módulo**. Si solo lo usa `training_core.py`, va en `training_core.py`.

---

### `serving_core.py` (Núcleo de Serving)

**Rol:** Gestionar **deployments** (modelos desplegados como endpoints). Contiene:
- `ServingStore` (CRUD de deployments + semáforos + pool).
- `EventLog` (JSONL de eventos).
- Generación de documentación API.
- Error recording.

---

### `training_core.py` (Núcleo de Entrenamiento)

**Rol:** Gestionar **jobs** de entrenamiento. Contiene:
- `TrainingStore` (CRUD de jobs + pool).
- Detección de framework fit (Keras adapter).
- Cross-validation integration.
- Artifact management (atomic swap).
- `_run_fit()` y `_run_evaluate()`.

---

### `monitoring_core.py` (Núcleo de Monitoreo)

**Rol:** **Calcular** métricas a partir de eventos. No **graba** eventos (eso lo hace `serving_core.py`). Contiene:
- `aggregate()` (la función principal de métricas).
- `range_window()` y `bucketize()`.
- `deployment_overview()`.

---

### `training_datasets.py` (Conjuntos de Datos)

**Rol:** Leer, validar y **guardar en RAM** los datasets. Contiene:
- `DatasetStore` (dict de DataFrames en RAM).
- Formato detection (CSV/XLSX/Parquet).
- URL fetching con IP validation (anti-SSRF).
- Column metadata y preview.

---

### `cross_validation.py` (Validación Cruzada)

**Rol:** Helpers **puros** (sin I/O) para K-fold CV. Contiene:
- `normalise_options()` (validar y normalizar params de CV).
- `is_better()` (comparar scores).
- Constantes (DEFAULT_FOLDS, MIN/MAX_FOLDS, DEFAULT_SEED).

**¿Por qué separado?** Porque `training_core.py` ya es enorme (2,900 líneas). Los helpers de CV son **pequeños** y **reutilizables** (podría usarlos otro módulo). También facilita tests unitarios sin importar `training_core`.

---

### `training_metrics.py` (Métricas de Referencia)

**Rol:** Implementaciones **puras** (sin TensorFlow) de métricas para **tests unitarios**. Contiene:
- `confusion_matrix()`.
- `cohens_kappa()`.
- `_as_labels()` (unwrap numpy scalars).

**¿Por qué sin TensorFlow?** TensorFlow es pesado y tiene side effects. Para un test unitario, solo necesitas la matemática. La implementación trainable de la métrica está en el sandbox (`_INVOKE_BOOTSTRAP`), no aquí.

---

### `notebook_generator.py` (Generador de Notebooks)

**Rol:** Generar notebooks Jupyter a partir del catálogo. Contiene:
- `_split_parameters()` (tunable vs fixed).
- `_is_safe_literal()` (AST check).
- `_widget_value()` (valor por defecto del widget).
- Cell construction y validación.

**Principio:** **Nunca** se escribe un notebook a mano. Si se genera, no se puede desactualizar.

---

### `server_fastapi.py` (Punto de Entrada HTTP)

**Rol:** Crear el `FastAPI()` app, registrar middlewares, routers, y endpoints principales. Contiene:
- `ServerSettings`.
- `ServerState`.
- `_AliasedBody`.
- Endpoints: models, sessions, health, metrics.

---

### `serving_api.py` (Rutas de Serving)

**Rol:** Routers de FastAPI para deployments, predict, monitoring. Contiene:
- `deployment_payload()`.
- `_stats_by_endpoint()`.
- Endpoints: deploy, undeploy, predict, events, monitoring.

---

### `training_api.py` (Rutas de Entrenamiento)

**Rol:** Routers de FastAPI para jobs, datasets, series analysis. Contina:
- `_job_payload()`.
- `SeriesAnalysisRequest`.
- Endpoints: submit job, job detail, dataset upload, series analysis.

---

*Fin del glosario.*
