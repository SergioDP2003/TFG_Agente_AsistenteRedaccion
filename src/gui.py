"""
Interfaz web (Gradio) para el agente de LangGraph definido en app1.py.

Este módulo no contiene lógica de negocio del grafo: reutiliza el grafo ya
compilado (`app`), el constructor de estado inicial (`crear_estado_inicial`),
las funciones de guardado (`guardar_state`, `guardar_documento_latex`) y la
configuración de los LLM (`PROVEEDORES_LLM`, `cargar_configuracion_llms`,
`guardar_configuracion_llms`) de `app1.py`. Su responsabilidad es triple:
  1. Traducir cada pausa por `interrupt()` del grafo en una interacción de
     chat: mostrar el historial acumulado y reanudar la ejecución con
     `Command(resume=...)` cuando el usuario envía texto.
  2. Ofrecer una página de "Ajustes" para gestionar gráficamente la carpeta
     `trabajos_relacionados/`, el archivo `src/.env` y los 2 modelos LLM
     ("simple" y "complejo") que usa el agente.
  3. Persistir esa configuración de modelos en `src/llm_config.json`.
"""

import os
import shutil
import uuid

import gradio as gr
from dotenv import dotenv_values, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_ENV = os.path.join(_SRC_DIR, ".env")
CARPETA_PDFS = "trabajos_relacionados"

# Cargamos el .env real (si existe) antes de importar app1, para que
# `crear_llm`/`LLMPerezoso` puedan resolver las API keys en su primer uso real.
# A diferencia de la versión anterior de este fichero, ya NO hace falta rellenar
# variables de relleno para evitar un fallo en el import: app1.py construye los
# modelos de forma perezosa (ver `LLMPerezoso`), así que importar el módulo es
# siempre seguro, exista o no exista `src/.env` todavía.
if os.path.exists(RUTA_ENV):
    load_dotenv(RUTA_ENV)

from app1 import (  # noqa: E402
    app,
    crear_estado_inicial,
    guardar_state,
    guardar_documento_latex,
    PROVEEDORES_LLM,
    cargar_configuracion_llms,
    guardar_configuracion_llms,
)

CLAVES_ENV = sorted({info["api_key_env"] for info in PROVEEDORES_LLM.values() if info["api_key_env"]})

MENSAJE_FINALIZADO = (
    "✅ Proceso completado. Se han guardado `src/state_guardado.json` y `related_works.tex`."
)

DESCRIPCION_INICIO = """
Este asistente entrevista sobre tu propio trabajo y analiza los PDFs que
coloques en `trabajos_relacionados/` para redactar automáticamente la sección
**Related Works** de tu paper: introducción, desarrollo por trabajo (o por
categorías), tabla comparativa opcional y conclusión — todo listo para pegar
en LaTeX con sus `\\cite{}` y la bibliografía ya generada.
"""


# ---- Historial de chat ----

def mensajes_a_historial(mensajes):
    """Convierte AgentState['messages'] al formato de mensajes que espera gr.Chatbot."""
    historial = []
    for msg in mensajes:
        if isinstance(msg, HumanMessage):
            historial.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            historial.append({"role": "assistant", "content": msg.content})
    return historial


def nueva_configuracion():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def ejecutar_hasta_pausa(entrada, config):
    """
    Generador que va emitiendo (historial_chat, terminado) a medida que el grafo
    produce mensajes, hasta que se pausa en un interrupt() o llega a END.

    NOTA: la condición de parada se decide mirando si la clave "__interrupt__"
    aparece en el último chunk del propio stream, y no con
    `app.get_state(config).next`: cuando un nodo lanza más de un interrupt()
    distinto en la misma ejecución (p.ej. el menú "modificar" de categorías o de
    la tabla), ese `.next` deja de ser fiable justo después del primer resume,
    aunque el grafo siga correctamente pausado.
    """
    ultimo_chunk = {}
    for chunk in app.stream(entrada, config, stream_mode="values"):
        ultimo_chunk = chunk
        if "__interrupt__" not in chunk and "messages" in chunk:
            yield mensajes_a_historial(chunk["messages"]), False

    if "__interrupt__" in ultimo_chunk:
        # El grafo está pausado en un interrupt() esperando entrada del usuario.
        valor = ultimo_chunk["__interrupt__"][0].value
        historial = mensajes_a_historial(ultimo_chunk.get("messages", []))
        if valor:
            historial.append({"role": "assistant", "content": str(valor)})
        yield historial, False
        return

    # El grafo ha llegado a END: preservamos los mismos outputs que la consola.
    guardar_state(ultimo_chunk, "state_guardado.json")
    guardar_documento_latex(ultimo_chunk, "related_works.tex")
    historial = mensajes_a_historial(ultimo_chunk.get("messages", []))
    historial.append({"role": "assistant", "content": MENSAJE_FINALIZADO})
    yield historial, True


def iniciar_conversacion():
    config = nueva_configuracion()
    terminado = False
    for historial, terminado in ejecutar_hasta_pausa(crear_estado_inicial(), config):
        yield historial, config, gr.update()
    yield historial, config, gr.update(
        interactive=not terminado,
        placeholder="Proceso finalizado." if terminado else "Escribe aquí tu respuesta...",
    )


def manejar_envio(mensaje, config):
    if not config or not mensaje or not mensaje.strip():
        yield gr.update(), gr.update(value="")
        return

    historial = []
    terminado = False
    for historial, terminado in ejecutar_hasta_pausa(Command(resume=mensaje), config):
        yield historial, gr.update(value="")
    yield historial, gr.update(
        value="",
        interactive=not terminado,
        placeholder="Proceso finalizado." if terminado else "Escribe aquí tu respuesta...",
    )


# ---- Página de ajustes: PDFs ----

def _listar_pdfs():
    os.makedirs(CARPETA_PDFS, exist_ok=True)
    return sorted(f for f in os.listdir(CARPETA_PDFS) if f.lower().endswith(".pdf"))


def refrescar_lista_pdfs():
    return gr.update(choices=_listar_pdfs(), value=[])


def anadir_pdfs(archivos):
    if not archivos:
        return gr.update(choices=_listar_pdfs()), gr.update(), None

    os.makedirs(CARPETA_PDFS, exist_ok=True)
    copiados = []
    for ruta in archivos:
        destino = os.path.join(CARPETA_PDFS, os.path.basename(ruta))
        shutil.copy(ruta, destino)
        copiados.append(os.path.basename(ruta))

    mensaje = f"✅ Añadido(s) {len(copiados)} PDF(s): {', '.join(copiados)}"
    return gr.update(choices=_listar_pdfs(), value=[]), mensaje, None


def eliminar_pdfs(seleccionados):
    if not seleccionados:
        return gr.update(choices=_listar_pdfs(), value=[]), "⚠️ No se ha seleccionado ningún PDF para eliminar."

    for nombre in seleccionados:
        ruta = os.path.join(CARPETA_PDFS, nombre)
        if os.path.exists(ruta):
            os.remove(ruta)

    return gr.update(choices=_listar_pdfs(), value=[]), f"🗑️ Eliminado(s): {', '.join(seleccionados)}"


# ---- Página de ajustes: src/.env ----

def cargar_valores_env():
    valores = dotenv_values(RUTA_ENV) if os.path.exists(RUTA_ENV) else {}
    return [valores.get(clave, "") or "" for clave in CLAVES_ENV]


def guardar_env(*valores):
    contenido = "\n".join(f"{clave}={valor}" for clave, valor in zip(CLAVES_ENV, valores)) + "\n"
    with open(RUTA_ENV, "w", encoding="utf-8") as f:
        f.write(contenido)

    # Reflejamos también los valores en el proceso actual: si el LLM correspondiente
    # todavía no se ha resuelto (ver LLMPerezoso en app1.py), la clave nueva se
    # recogerá sin necesidad de reiniciar la aplicación.
    for clave, valor in zip(CLAVES_ENV, valores):
        if valor:
            os.environ[clave] = valor

    return (
        f"✅ Variables guardadas en `src/.env` ({', '.join(CLAVES_ENV)}). "
        "Si algún modelo ya se había usado en esta sesión, reinicia la aplicación "
        "(`python src/gui.py`) para que recoja la clave nueva."
    )


# ---- Página de ajustes: modelos LLM ----

OPCIONES_PROVEEDOR = [(info["etiqueta"], clave) for clave, info in PROVEEDORES_LLM.items()]


def cargar_valores_modelos_llm():
    config = cargar_configuracion_llms()
    simple = config["simple"]
    complejo = config["complejo"]
    return (
        simple.get("proveedor", "ollama"),
        simple.get("modelo", ""),
        simple.get("temperature", 0.1),
        complejo.get("proveedor", "google_genai"),
        complejo.get("modelo", ""),
        complejo.get("temperature", 0.3),
    )


def guardar_modelos_llm(proveedor_simple, modelo_simple, temp_simple, proveedor_complejo, modelo_complejo, temp_complejo):
    config = {
        "simple": {"proveedor": proveedor_simple, "modelo": modelo_simple.strip(), "temperature": temp_simple},
        "complejo": {"proveedor": proveedor_complejo, "modelo": modelo_complejo.strip(), "temperature": temp_complejo},
    }
    guardar_configuracion_llms(config)
    return (
        "✅ Configuración de modelos guardada en `src/llm_config.json`. "
        "Si esta sesión ya había ejecutado el agente, reinicia la aplicación "
        "(`python src/gui.py`) para que los nuevos modelos se apliquen."
    )


# ---- Navegación entre páginas ----

def mostrar_inicio():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


def mostrar_ajustes():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


def mostrar_chat():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


with gr.Blocks(title="AI Related Works Agent") as demo:
    # ---- Página de inicio ----
    with gr.Column(visible=True) as pagina_inicio:
        gr.Markdown("# AI Related Works Agent")
        gr.Markdown(DESCRIPCION_INICIO)
        with gr.Row():
            comenzar_btn = gr.Button("▶️ Comenzar", variant="primary", scale=1)
            ajustes_btn = gr.Button("⚙️ Ajustes", scale=1)

    # ---- Página de ajustes ----
    with gr.Column(visible=False) as pagina_ajustes:
        gr.Markdown("## ⚙️ Ajustes")

        gr.Markdown("### 📄 Documentos en `trabajos_relacionados/`")
        lista_pdfs = gr.CheckboxGroup(
            label="PDFs actuales (marca los que quieras eliminar)",
            choices=_listar_pdfs(),
        )
        with gr.Row():
            subir_pdfs = gr.File(
                label="Añadir nuevos PDFs",
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath",
            )
            eliminar_pdfs_btn = gr.Button("🗑️ Eliminar seleccionados")
        estado_pdfs_txt = gr.Markdown("")

        gr.Markdown("### 🧠 Modelos LLM")
        gr.Markdown(
            "El agente usa 2 modelos: uno para tareas **simples** (extracción, evaluación, "
            "párrafos cortos) y otro para tareas **complejas** (propuesta de categorías, "
            "redacción del cuerpo, generación de la tabla y ensamblado final). Cada uno puede "
            "ser un modelo local de Ollama o un servicio externo mediante API key — pueden ser "
            "iguales o distintos. Si eliges un proveedor de tipo API, recuerda rellenar su clave "
            "en la sección de variables de entorno de abajo."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Modelo para tareas simples**")
                proveedor_simple_dd = gr.Dropdown(
                    label="Proveedor", choices=OPCIONES_PROVEEDOR, value="ollama"
                )
                modelo_simple_txt = gr.Textbox(label="Nombre del modelo", placeholder="ej: llama3.2:3b")
                temp_simple_sl = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.05, value=0.1)
            with gr.Column():
                gr.Markdown("**Modelo para tareas complejas**")
                proveedor_complejo_dd = gr.Dropdown(
                    label="Proveedor", choices=OPCIONES_PROVEEDOR, value="google_genai"
                )
                modelo_complejo_txt = gr.Textbox(label="Nombre del modelo", placeholder="ej: gemini-2.5-flash")
                temp_complejo_sl = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.05, value=0.3)
        guardar_modelos_btn = gr.Button("💾 Guardar modelos LLM", variant="primary")
        estado_modelos_txt = gr.Markdown("")

        gr.Markdown("### 🔑 Variables de entorno (`src/.env`)")
        gr.Markdown(
            "Claves de API necesarias según los proveedores elegidos arriba. "
            "Si `src/.env` todavía no existe, se creará al guardar."
        )
        env_textboxes = {}
        for _clave in CLAVES_ENV:
            env_textboxes[_clave] = gr.Textbox(
                label=_clave,
                type="password",
                placeholder=f"Introduce el valor de {_clave}",
            )
        guardar_env_btn = gr.Button("💾 Guardar variables de entorno", variant="primary")
        estado_env_txt = gr.Markdown("")

        volver_btn = gr.Button("← Volver al inicio")

    # ---- Página de la conversación ----
    with gr.Column(visible=False) as pagina_chat:
        chatbot = gr.Chatbot(height=600, label="Conversación")
        config_state = gr.State(None)

        with gr.Row():
            entrada_txt = gr.Textbox(
                placeholder="Escribe aquí tu respuesta...",
                show_label=False,
                scale=8,
            )
            enviar_btn = gr.Button("Enviar", scale=1, variant="primary")

    # ---- Cableado de navegación ----
    comenzar_btn.click(
        fn=mostrar_chat, outputs=[pagina_inicio, pagina_ajustes, pagina_chat]
    ).then(
        fn=iniciar_conversacion, outputs=[chatbot, config_state, entrada_txt]
    )

    ajustes_btn.click(
        fn=mostrar_ajustes, outputs=[pagina_inicio, pagina_ajustes, pagina_chat]
    ).then(
        fn=refrescar_lista_pdfs, outputs=[lista_pdfs]
    ).then(
        fn=cargar_valores_env, outputs=list(env_textboxes.values())
    ).then(
        fn=cargar_valores_modelos_llm,
        outputs=[
            proveedor_simple_dd, modelo_simple_txt, temp_simple_sl,
            proveedor_complejo_dd, modelo_complejo_txt, temp_complejo_sl,
        ],
    )

    volver_btn.click(fn=mostrar_inicio, outputs=[pagina_inicio, pagina_ajustes, pagina_chat])

    # ---- Cableado de ajustes ----
    subir_pdfs.upload(fn=anadir_pdfs, inputs=[subir_pdfs], outputs=[lista_pdfs, estado_pdfs_txt, subir_pdfs])
    eliminar_pdfs_btn.click(fn=eliminar_pdfs, inputs=[lista_pdfs], outputs=[lista_pdfs, estado_pdfs_txt])
    guardar_env_btn.click(fn=guardar_env, inputs=list(env_textboxes.values()), outputs=[estado_env_txt])
    guardar_modelos_btn.click(
        fn=guardar_modelos_llm,
        inputs=[
            proveedor_simple_dd, modelo_simple_txt, temp_simple_sl,
            proveedor_complejo_dd, modelo_complejo_txt, temp_complejo_sl,
        ],
        outputs=[estado_modelos_txt],
    )

    # ---- Cableado del chat ----
    enviar_btn.click(
        fn=manejar_envio,
        inputs=[entrada_txt, config_state],
        outputs=[chatbot, entrada_txt],
    )
    entrada_txt.submit(
        fn=manejar_envio,
        inputs=[entrada_txt, config_state],
        outputs=[chatbot, entrada_txt],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
