"""
Interfaz web (Gradio) para el agente de LangGraph definido en app1.py.

Este módulo no contiene lógica de negocio: reutiliza el grafo ya compilado
(`app`), el constructor de estado inicial (`crear_estado_inicial`) y las
funciones de guardado (`guardar_state`, `guardar_documento_latex`) de
`app1.py`. Su única responsabilidad es traducir cada pausa por `interrupt()`
del grafo en una interacción de chat: mostrar el historial acumulado y
reanudar la ejecución con `Command(resume=...)` cuando el usuario envía texto.
"""

import uuid

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from app1 import app, crear_estado_inicial, guardar_state, guardar_documento_latex

MENSAJE_FINALIZADO = (
    "✅ Proceso completado. Se han guardado `src/state_guardado.json` y `related_works.tex`."
)


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


with gr.Blocks(title="AI Related Works Agent") as demo:
    gr.Markdown(
        "# AI Related Works Agent\n"
        "Redacta la sección *Related Works* de tu paper de forma interactiva, "
        "a partir de los PDFs en `trabajos_relacionados/`."
    )

    chatbot = gr.Chatbot(height=600, label="Conversación")
    config_state = gr.State(None)

    with gr.Row():
        entrada_txt = gr.Textbox(
            placeholder="Escribe aquí tu respuesta...",
            show_label=False,
            scale=8,
        )
        enviar_btn = gr.Button("Enviar", scale=1, variant="primary")

    demo.load(fn=iniciar_conversacion, outputs=[chatbot, config_state, entrada_txt])

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
