import os
import json
import re

from typing import Dict, TypedDict, List, Union, Annotated, Sequence, Optional
from pypdf import PdfReader
from collections import defaultdict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

load_dotenv()

# ------------------------------- CONFIGURACIÓN DE LOS LLM ----------------------------------
#
# El agente usa 2 "slots" de modelo configurables desde la página de Ajustes de la GUI:
#   - "simple":   tareas menos exigentes (extracción, evaluación, párrafos cortos).
#   - "complejo": tareas más exigentes (propuesta de categorías, redacción del cuerpo,
#                 generación de la tabla, ensamblado final en LaTeX).
# Cada slot puede apuntar a un modelo local de Ollama o a un servicio externo por API key.
# `init_chat_model` (de LangChain) resuelve la clase concreta (ChatOllama,
# ChatGoogleGenerativeAI, ChatOpenAI, ChatAnthropic...) a partir del nombre de proveedor,
# siempre que el paquete langchain-<proveedor> correspondiente esté instalado.

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
RUTA_LLM_CONFIG = os.path.join(_SRC_DIR, "llm_config.json")
CARPETA_PDFS = os.path.join(_REPO_ROOT, "trabajos_relacionados")

PROVEEDORES_LLM = {
    "ollama": {
        "etiqueta": "Ollama (modelo local)",
        "api_key_env": None,
        "modelo_defecto": "llama3.2:3b",
    },
    "google_genai": {
        "etiqueta": "Google Gemini (API)",
        "api_key_env": "GOOGLE_API_KEY",
        "modelo_defecto": "gemini-2.5-flash",
    },
    "openai": {
        "etiqueta": "OpenAI (API)",
        "api_key_env": "OPENAI_API_KEY",
        "modelo_defecto": "gpt-4o-mini",
    },
    "anthropic": {
        "etiqueta": "Anthropic Claude (API)",
        "api_key_env": "ANTHROPIC_API_KEY",
        "modelo_defecto": "claude-3-5-haiku-latest",
    },
}

# Configuración de fábrica: reproduce el comportamiento previo a que esto fuera
# configurable (llama3.2:3b local para tareas simples, Gemini para las complejas).
CONFIG_LLM_DEFECTO = {
    "simple": {"proveedor": "ollama", "modelo": "llama3.2:3b", "temperature": 0.1},
    "complejo": {"proveedor": "google_genai", "modelo": "gemini-2.5-flash", "temperature": 0.3},
}

# Nº máximo de caracteres de cada PDF que `analizar_trabajos_node` envía al LLM "simple"
# (ver más abajo). 0 o negativo = sin límite: se envía el texto completo de cada PDF, tal
# cual, sin importar lo corto o largo que sea (el slicing de Python no falla si el texto es
# más corto que el límite, así que no hay caso especial que romperse con PDFs pequeños).
LIMITE_CARACTERES_ANALISIS_DEFECTO = 12000


def cargar_configuracion_llms() -> dict:
    """Lee src/llm_config.json (creado/editado desde la página de Ajustes de la GUI).
    Si el fichero no existe todavía o está corrupto, se usa la configuración de fábrica."""
    if os.path.exists(RUTA_LLM_CONFIG):
        try:
            with open(RUTA_LLM_CONFIG, "r", encoding="utf-8") as f:
                datos = json.load(f)
            return {
                "simple": {**CONFIG_LLM_DEFECTO["simple"], **datos.get("simple", {})},
                "complejo": {**CONFIG_LLM_DEFECTO["complejo"], **datos.get("complejo", {})},
                "limite_caracteres_analisis": datos.get(
                    "limite_caracteres_analisis", LIMITE_CARACTERES_ANALISIS_DEFECTO
                ),
            }
        except (json.JSONDecodeError, OSError):
            pass
    return {
        **{slot: dict(cfg) for slot, cfg in CONFIG_LLM_DEFECTO.items()},
        "limite_caracteres_analisis": LIMITE_CARACTERES_ANALISIS_DEFECTO,
    }


def guardar_configuracion_llms(config: dict):
    """Escribe src/llm_config.json desde cero (lo llama la página de Ajustes de la GUI)."""
    with open(RUTA_LLM_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def crear_llm(config_slot: dict):
    """Instancia un chat model a partir de un slot de configuración
    ({"proveedor", "modelo", "temperature"}) usando `init_chat_model`."""
    proveedor = config_slot.get("proveedor", "ollama")
    info_proveedor = PROVEEDORES_LLM.get(proveedor, PROVEEDORES_LLM["ollama"])

    kwargs = {
        "model": config_slot.get("modelo") or info_proveedor["modelo_defecto"],
        "model_provider": proveedor,
        "temperature": config_slot.get("temperature", 0.2),
    }

    clave_env = info_proveedor.get("api_key_env")
    if clave_env:
        valor = os.getenv(clave_env)
        if valor:
            kwargs["api_key"] = valor

    return init_chat_model(**kwargs)


class LLMPerezoso:
    """Envoltorio que retrasa la construcción real del modelo hasta el primer uso.

    Antes, `llm`/`llm3` se construían a nivel de módulo: si faltaba la API key de
    Gemini, el simple `import app1` ya lanzaba una excepción (ver README/CLAUDE.md).
    Con este envoltorio, un proveedor mal configurado (API key ausente, paquete no
    instalado, modelo inexistente) no impide importar el módulo: el error solo
    aparece cuando el nodo que de verdad invoca ese modelo se ejecuta. Además, como
    la configuración se relee de `llm_config.json` en el primer uso (no en el
    import), guardar cambios desde Ajustes puede surtir efecto sin reiniciar el
    proceso, siempre que ningún nodo haya usado ya ese slot en el proceso actual.
    """

    def __init__(self, slot: str):
        self._slot = slot
        self._modelo = None

    def _resolver(self):
        if self._modelo is None:
            config = cargar_configuracion_llms()[self._slot]
            self._modelo = crear_llm(config)
        return self._modelo

    def invoke(self, *args, **kwargs):
        return self._resolver().invoke(*args, **kwargs)

    def with_structured_output(self, *args, **kwargs):
        return self._resolver().with_structured_output(*args, **kwargs)


llm_simple = LLMPerezoso("simple")
llm_complejo = LLMPerezoso("complejo")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    idioma_salida: str

    tema_paper: str
    trabajos_pdf: List[str]
    textos_trabajos: List[str]
    trabajos_analizados: List[Dict]

    categorizar_activo: bool
    categorias_propuestas: str
    categorias_confirmadas: bool
    accion_categorias: str
    instrucciones_modificacion: Optional[str]
    introduccion_related_works: str
    related_works_section: str

    recomendacion_tabla: str
    tabla_comparativa_activa: bool
    estructura_tabla_propuesta: Dict
    estructura_tabla_confirmada: bool
    accion_estructura: str
    instrucciones_tabla: Optional[str]  
    tabla_comparativa_generada: str
    descripcion_tabla: str
    conclusion_related_work: str

    related_works_document: str

    error: Optional[str]

class PaperEstructurado(BaseModel):
    titulo: str = Field(description="Título ORIGINAL del paper. NO añadas frases como 'Ficha técnica' o 'Resumen'. Solo el texto del título.")
    autores: List[str] = Field(description="Lista de autores.")
    anio: str = Field(description="Año (4 dígitos).")
    problema_especifico: str = Field(description="¿Cuál es el vacío o limitación técnica exacta que motiva este estudio? No repitas el título.")
    metodologia_detallada: str = Field(description="Menciona componentes técnicos específicos: arquitecturas, módulos, algoritmos o frameworks usados.")
    aportaciones_clave: str = Field(description="¿Cuál es la novedad principal?")
    limitaciones_criticas: str = Field(description="Debilidades técnicas mencionadas o inferidas.")
    casos_uso: List[str] = Field(description="Lista de aplicaciones.")

class Categoria(BaseModel):
    nombre: str = Field(description="Nombre corto de la categoría")
    descripcion: str = Field(description="Justificación técnica de la categoría")
    trabajos: List[str] = Field(description="Lista de TÍTULOS de los papers que van aquí")

class PropuestaCategorias(BaseModel):
    categorias: List[Categoria] = Field(description="Lista de máximo 3 categorías en base a los trabajos existentes.")

# -------------------------------- METODOS AUXILIARES -------------------------------------

def _truncar_texto(texto: str, limite: int) -> str:
    """Recorta `texto` a `limite` caracteres y añade "..." SOLO si de verdad se ha recortado algo.
    Evita el bug de mostrar "..." tras un texto que ya cabía entero, lo que hacía parecer cortados
    títulos que en realidad estaban completos.
    """
    texto = texto or ""
    return texto if len(texto) <= limite else texto[:limite].rstrip() + "..."

def extraer_json_puro(texto: str):
    """Extrae el JSON eliminando cualquier texto extra del LLM."""
    try:
        # Busca lo que esté entre llaves
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None
    except Exception:
        return None
    
def guardar_state(state: dict, archivo: str = "state_guardado_1.json"):

    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_completa = os.path.join(directorio_actual, archivo)

    state_serializable = state.copy()
    mensajes_serializados = []
    
    for msg in state.get("messages", []):
        mensajes_serializados.append({
            "type": msg.__class__.__name__,
            "content": msg.content
        })
    
    state_serializable["messages"] = mensajes_serializados

    with open(ruta_completa, "w", encoding="utf-8") as f:
        json.dump(state_serializable, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Estado guardado en la ruta del script: {ruta_completa}")

def limpiar_texto_academico(texto: str):

    texto = re.sub(r'(?<=[A-Z])\s(?=[A-Z])', '', texto)
    return " ".join(texto.split())

def extraer_json(texto: str):
    try:
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            return None
    except Exception:
        return None

def _normalizar_columnas(estructura):
    """Red de seguridad: todo el pipeline de la tabla (estructuras_similares, mensajes al
    usuario, generar_tabla_node) espera que "columnas" sea una lista plana de strings, pero un
    LLM puede devolver en su lugar objetos tipo {"nombre": "...", "tipo": "descriptiva"} —
    especialmente en proponer_estructura_node, cuyo prompt le pide razonar explícitamente sobre
    el tipo de cada columna. Reducimos cada elemento a su nombre para blindar el resto del flujo.
    """
    if not estructura or "columnas" not in estructura:
        return estructura

    columnas_normalizadas = []
    for c in estructura.get("columnas", []):
        if isinstance(c, dict):
            nombre = c.get("nombre") or c.get("name") or c.get("columna") or next(iter(c.values()), "")
            columnas_normalizadas.append(str(nombre))
        else:
            columnas_normalizadas.append(str(c))
    estructura["columnas"] = columnas_normalizadas
    return estructura

def estructuras_similares(e1, e2, umbral=0.7):
    if not e1 or not e2:
        return False

    cols1 = set([c.lower() for c in e1.get("columnas", [])])
    cols2 = set([c.lower() for c in e2.get("columnas", [])])

    if not cols1 or not cols2:
        return False

    inter = len(cols1.intersection(cols2))
    union = len(cols1.union(cols2))

    similitud = inter / union

    return similitud >= umbral

def guardar_documento_latex(state: dict, nombre_archivo: str = "related_works.tex"):
    """
    Extrae el contenido de 'related_works_document' del estado del agente
    y lo guarda en un archivo con extensión .tex en formato UTF-8.
    """
    # 1. Extraemos el texto en LaTeX del diccionario de estado
    codigo_latex = state.get("related_works_document", "")
    
    if not codigo_latex:
        print("⚠️ No se encontró contenido en 'related_works_document' para guardar.")
        return False
        
    try:
        # 2. Abrimos y escribimos el archivo asegurando codificación UTF-8
        with open(nombre_archivo, "w", encoding="utf-8") as archivo:
            archivo.write(codigo_latex)
            
        print(f"💾 ¡Archivo LaTeX guardado con éxito! -> {nombre_archivo}")
        return True
        
    except Exception as e:
        print(f"❌ Error al guardar el archivo .tex: {e}")
        return False

# ------------------------------------ NODOS --------------------------------------------------

def seleccionar_idioma_node(state: AgentState) -> AgentState:
    """
    Nodo 0: Seleccionar Idioma de Salida (Tarea de Usuario - Interactivo)
    Anclado al inicio del grafo para definir el idioma global del manuscrito.
    """
    # Diccionario de mapeo rápido para estandarizar la entrada
    mapa_idiomas = {
        "1": "Español académico",
        "2": "Inglés académico (English)"
    }

    # El menú de idiomas ya se muestra en el mensaje de bienvenida inicial,
    # así que no repetimos el texto aquí salvo que la opción sea inválida.
    mensaje_interrupcion = ""

    while True:

        opcion = interrupt(mensaje_interrupcion).strip()

        if opcion in mapa_idiomas:
            idioma_elegido = mapa_idiomas[opcion]
            
            # Mensajes en primera persona para el feed conversacional
            texto_humano = f"Opción {opcion}: Prefiero el documento final en {idioma_elegido}."
            texto_agente = f"Idioma global fijado en {idioma_elegido}. Toda la suite de redacción y la compilación final de LaTeX se ejecutarán bajo esta directriz."
            texto_agente2 = f"Se procede con la definición del paper desarrollado. Debes describir detalladamente de qué trata tu paper, su metodología y qué aporta (Ej: Un framework llamado SimulateIoT-Services...)"

            return {
                "idioma_salida": idioma_elegido,  # Nueva clave para tu AgentState
                "error": None,
                "messages": [
                    HumanMessage(content=texto_humano),
                    AIMessage(content=texto_agente),
                    AIMessage(content=texto_agente2)
                ]
            }
        else:
            mensaje_interrupcion = "⚠️ Opción inválida. Por favor, introduce un número del 1 al 2."

def definir_tematica_node(state: AgentState) -> AgentState:

    # Capturamos la descripción del usuario (el nodo anterior ya explica qué se le pide)
    tema_usuario = interrupt("")
    
    prompt = f"""
    Analiza la descripción del paper que está escribiendo el usuario. Tu tarea es extraer y rellenar la ficha técnica formal de SU propia investigación utilizando el esquema estructurado. Infere los aspectos técnicos basándote en su explicación.
    
    Descripción del usuario:
    "{tema_usuario}"
    """
    
    try:
        # Forzamos al LLM a escupir la estructura idéntica a la de los papers analizados
        llm_estructurado = llm_simple.with_structured_output(PaperEstructurado)
        res_pydantic = llm_estructurado.invoke(prompt)
        
        # Guardamos como diccionario estándar
        metadatos_dict = res_pydantic.model_dump()

        # --- FICHA TÉCNICA EN MARKDOWN (para que gr.Chatbot la renderice legible por campos, en
        # vez de un único bloque JSON en crudo) ---
        def _cita_multilinea(texto):
            lineas = (texto or "").strip().splitlines() or [""]
            return [f"> {linea}" if linea.strip() else ">" for linea in lineas]

        autores = metadatos_dict.get("autores") or []
        casos_uso = metadatos_dict.get("casos_uso") or []

        lineas_ficha = [
            "### 🗂️ Ficha técnica de tu paper",
            "",
            f"**Título:** {metadatos_dict.get('titulo', '')}",
            f"**Autores:** {', '.join(autores) if autores else 'No especificado'}",
            f"**Año:** {metadatos_dict.get('anio', '')}",
            "",
            "**Problema específico:**",
            "",
            *_cita_multilinea(metadatos_dict.get("problema_especifico")),
            "",
            "**Metodología:**",
            "",
            *_cita_multilinea(metadatos_dict.get("metodologia_detallada")),
            "",
            "**Aportaciones clave:**",
            "",
            *_cita_multilinea(metadatos_dict.get("aportaciones_clave")),
            "",
            "**Limitaciones críticas:**",
            "",
            *_cita_multilinea(metadatos_dict.get("limitaciones_criticas")),
            "",
            "**Casos de uso:**",
            "",
            *([f"- {c}" for c in casos_uso] if casos_uso else ["- No especificado"]),
        ]

        mensaje_agente = f"Temática del paper recibida correctamente. Realizando un análisis y estructuración de la información."
        mensaje_agente2 = "\n".join(lineas_ficha)
        mensaje_agente3 = f"Se procede a la búsqueda de los PDFs..."
        
        return {
            "tema_paper": metadatos_dict,
            "messages": [
                HumanMessage(content=f"Descripción inicial: {tema_usuario}"),
                AIMessage(content=mensaje_agente),
                AIMessage(content=mensaje_agente2),
                AIMessage(content=mensaje_agente3)
            ]
        }
        
    except Exception as e:
        print(f"⚠️ Error al estructurar la temática: {e}")
        # Fallback estructural seguro para que el grafo jamás se rompa si falla el parsing
        fallback_dict = {
            "titulo": "Propuesta de Investigación",
            "autores": ["El Autor"],
            "anio": "2026",
            "problema_especifico": tema_usuario,
            "metodologia_detallada": "Enfoque basado en agentes e inteligencia artificial.",
            "aportaciones_clave": "Automatización del proceso de desarrollo.",
            "limitaciones_criticas": "Dependencia del modelo de lenguaje base.",
            "casos_uso": ["Entornos académicos"]
        }
        return {
            "tema_paper": fallback_dict,
            "messages": [
                HumanMessage(content=f"Tema (Fallback Estructurado): {tema_usuario}"),
                AIMessage(content="No se pudo parsear el formato estructurado. Se activa la ficha técnica de emergencia.")
            ]
        }

def buscar_pdfs_node(state: AgentState):
    carpeta = CARPETA_PDFS
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
        mensaje_error = (
            f"Alerta del sistema: La carpeta '{carpeta}' no existía y ha sido creada automáticamente.\n"
            f"Por favor, añade los PDFs de los trabajos relacionados que deseas analizar en ella y vuelve a ejecutar el agente."
        )
        return {
            "error": f"Crea la carpeta '{carpeta}' y añade los PDFs.", 
            "messages": [AIMessage(content=mensaje_error)]
        }
    
    archivos = [os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.endswith(".pdf")]
    num_archivos = len(archivos)
    if num_archivos == 0:
        mensaje_salida = (
            f"Exploración completada: Se localizó la carpeta '{carpeta}', pero está completamente vacía.\n"
            f"Asegúrate de arrastrar tus documentos científicos (.pdf) a esa ruta para poder proceder con el análisis. Se debe reiniciar el agente"
        )
        mensajes = [AIMessage(content=mensaje_salida)]
    else:
        # Generamos una lista visual de los ficheros encontrados para que el usuario sepa cuáles se van a leer
        lista_ficheros = "\n".join([f"   📄 - {os.path.basename(f)}" for f in archivos])
        mensaje_salida = (
            f"Exploración completada con éxito. Se han indexado {num_archivos} documentos para su lectura:\n"
            f"{lista_ficheros}\n\n"
        )

        mensaje_salida2 = f"Procediendo a la extracción de texto y metadatos..."
        mensajes = [AIMessage(content=mensaje_salida), AIMessage(content=mensaje_salida2)]

    return {
        "trabajos_pdf": archivos,
        "messages": mensajes
    }

def leer_texto_node(state: AgentState):
    textos = []
    archivos_procesados = []
    errores = []
    
    # Recorremos las rutas indexadas en el nodo anterior
    for ruta in state.get("trabajos_pdf", []):
        nombre_archivo = os.path.basename(ruta)
        try:
            reader = PdfReader(ruta)
            # Extraemos el texto de cada página de manera eficiente
            contenido = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            textos.append(contenido.strip())
            archivos_procesados.append(nombre_archivo)
        except Exception as e:
            errores.append(f"❌ {nombre_archivo}: {str(e)}")
            print(f"⚠️ Error interno leyendo {nombre_archivo}: {e}")

    # --- CONSTRUCCIÓN DEL MENSAJE CONVERSACIONAL ---
    lineas_reporte = []
    
    if archivos_procesados:
        lineas_reporte.append("📖 Extracción de texto completada en los siguientes manuscritos:")
        for archivo in archivos_procesados:
            lineas_reporte.append(f"   ✅ {archivo}")
            
    if errores:
        lineas_reporte.append("\n⚠️ Se presentaron inconvenientes con algunos archivos:")
        for err in errores:
            lineas_reporte.append(f"   {err}")
            
    if not textos:
        mensaje_final = (
            "Error en el proceso: No se pudo extraer texto de ningún PDF.\n"
            "Asegúrate de que los archivos no estén corruptos o protegidos contra lectura."
        )
    else:
        mensaje_final = (
            "\n".join(lineas_reporte) + 
            f"\n\nMemoria de texto cargada correctamente ({len(textos)} fuentes listas).\n"
        )

    mensaje_salida = f"Procediendo al análisis y estructuración de los datos científicos..."

    return {
        "textos_trabajos": textos, 
        "messages": [
            AIMessage(content=mensaje_final),
            AIMessage(content=mensaje_salida)
        ] 
    }

def analizar_trabajos_node(state: AgentState):
    analizados = []
    reporte_titulos = []
    
    structured_llm = llm_simple.with_structured_output(PaperEstructurado)

    # Configurable desde Ajustes: cuántos caracteres de cada PDF se envían al LLM simple.
    # 0 o negativo = sin límite (texto completo del PDF, para modelos con ventana de
    # contexto muy grande). El slicing `texto[:limite]` no falla si `texto` es más corto
    # que `limite` — simplemente devuelve el texto completo, así que un PDF con poco
    # contenido nunca rompe esto aunque el límite configurado sea enorme.
    limite_caracteres = cargar_configuracion_llms().get(
        "limite_caracteres_analisis", LIMITE_CARACTERES_ANALISIS_DEFECTO
    )

    for i, texto in enumerate(state["textos_trabajos"]):
        fragmento = texto if limite_caracteres <= 0 else texto[:limite_caracteres]

        print(f"🔄 Extrayendo datos únicos del trabajo {i+1}...")
        
        prompt = f"""
Extrae la ficha técnica del paper. 

REGLAS DE CALIDAD:
- TÍTULO: Extrae solo el nombre del paper, limpio.
- PROBLEMA: Define el 'Research Gap' (ej: 'Incapacidad de X para lograr Y'). 
- EVITA RELLENO: No uses frases como 'El trabajo utiliza...', ve directo al grano técnico.
- IDIOMA: Responde siempre en Español.

TEXTO:
{fragmento}
"""
        
        try:
            res = structured_llm.invoke(prompt)  
            data = res.model_dump()

            if "falta de memoria a largo plazo" in data["problema_especifico"].lower():
                 print(f"⚠️ Aviso: Posible sesgo en el problema del trabajo {i+1}")
            
            analizados.append(data)
            reporte_titulos.append(f"   🔹 [{i+1}] {_truncar_texto(res.titulo, 60)}")
            print(f"✅ FINALIZADO: {_truncar_texto(res.titulo, 50)}")
        except Exception as e:
            print(f"❌ Error en trabajo {i+1}: {e}")
    
    lista_papers_analizados = "\n".join(reporte_titulos)
    mensaje_salida = (
        f"📋 Extracción y estructuración de la literatura completada.\n"
        f"Se han generado fichas técnicas estandarizadas para los siguientes artículos:\n"
        f"{lista_papers_analizados}\n" 
    )

    # --- FICHAS TÉCNICAS EN MARKDOWN (para que gr.Chatbot las renderice legibles por campos, en
    # vez del `repr()` en crudo de la lista de diccionarios) ---
    def _cita_multilinea(texto):
        lineas = (texto or "").strip().splitlines() or [""]
        return [f"> {linea}" if linea.strip() else ">" for linea in lineas]

    fichas_bloques = []
    for idx, data in enumerate(analizados, 1):
        autores = data.get("autores") or []
        casos_uso = data.get("casos_uso") or []
        lineas_ficha = [
            f"#### {idx}. {data.get('titulo', '')}",
            "",
            f"**Autores:** {', '.join(autores) if autores else 'No especificado'}",
            f"**Año:** {data.get('anio', '')}",
            "",
            "**Problema específico:**",
            "",
            *_cita_multilinea(data.get("problema_especifico")),
            "",
            "**Metodología:**",
            "",
            *_cita_multilinea(data.get("metodologia_detallada")),
            "",
            "**Aportaciones clave:**",
            "",
            *_cita_multilinea(data.get("aportaciones_clave")),
            "",
            "**Limitaciones críticas:**",
            "",
            *_cita_multilinea(data.get("limitaciones_criticas")),
            "",
            "**Casos de uso:**",
            "",
            *([f"- {c}" for c in casos_uso] if casos_uso else ["- No especificado"]),
        ]
        fichas_bloques.append("\n".join(lineas_ficha))

    mensaje_salida2 = (
        "### 🗂️ Fichas técnicas de los trabajos analizados\n\n"
        + "\n\n---\n\n".join(fichas_bloques)
    )

    mensaje_salida3 = f"Avanzando al diseño de la sección de categorías..."
            
    return {
        "trabajos_analizados": analizados,
        "messages": [
            AIMessage(content=mensaje_salida),
            AIMessage(content=mensaje_salida2)
        ]
    }

def evaluar_categorizacion_node(state: AgentState) -> AgentState:

    trabajos = state.get("trabajos_analizados", [])

    if not trabajos:
        return {
            "error": "No hay trabajos analizados.",
            "messages": [AIMessage(content="No hay trabajos para evaluar.")]
        }

    prompt = f"""
    Tienes los siguientes trabajos analizados en formato estructurado:

    {json.dumps(trabajos, indent=2, ensure_ascii=False)}

    Evalúa si es recomendable categorizarlos.

    Responde con Sí o No junto con una breve explicación del por qué. La explicación de máximo 1 párrafo de 50 palabras.
    """

    response = llm_simple.invoke(prompt)

    mensaje_salida = (
        f"¿Te recomiendo añadir una división por categorías?\n"
        f"{response.content}\n"
        f"¿Cuál es tu decisión? (s/n)\n"
    )

    return {
        "messages": [AIMessage(content=mensaje_salida)]
    }
    
def decision_categorizacion_node(state: AgentState) -> AgentState:
    """
    Nodo: Decisión de categorización
    Tipo: Tarea Usuario
    Descripción: El usuario decide si categorizar o no los trabajos en base a la recomendación
    que ha realizado el agente tras el análisis de los trabajos.
    """

    # interrupt() debe quedar FUERA del try/except: internamente se implementa
    # lanzando una excepción para pausar el grafo, y un `except Exception` la
    # capturaría como si fuera un error real, saltándose la pausa por completo.
    decision = interrupt("")

    try:

        decision = decision.strip().lower()

        if decision not in ["s", "n"]:
            return {
                "error": "Debes responder 's' o 'n'.",
                "messages": [
                    AIMessage(content="La respuesta proporcionada no es válida. Debes responder 's' (sí) o 'n' (no).")
                ]
            }

        categorizar = (decision == "s")

        texto_humano = "Sí, prefiero organizar la sección 'Related Works' dividida por categorías temáticas." if categorizar else "No, prefiero una redacción continua de los trabajos sin divisiones temáticas."
        
        texto_agente = (
            "Decisión registrada con éxito. Iniciando la generación de propuestas de categorización..."
            if categorizar else 
            "Entendido. Omitiremos la creación de categorías y procederemos directamente con el diseño macro de la sección."
        )

        return {
            "categorizar_activo": categorizar,
            "error": None,
            "messages": [
                HumanMessage(content=texto_humano),
                AIMessage(content=texto_agente)
            ]
        }

    except Exception as e:
        print(f"⚠️ Error en la entrada de datos: {e}")
        return {
            "error": f"Error en decisión de categorización: {str(e)}",
            "messages": [
                AIMessage(content=f"🚨 Se interrumpió el flujo debido a un error inesperado en la entrada de datos: {str(e)}")
            ]
        }

def gateway_categorizacion(state: AgentState) -> str:
    if state.get("categorizar_activo"):
        return "proponer_categorias"
    else:
        return "redactar_introduccion"

def proponer_categorias_node(state: AgentState) -> AgentState:
    trabajos = state.get("trabajos_analizados", [])
    tema = state.get("tema_paper", "")

    titulos_reales = [t["titulo"] for t in trabajos]

    # Es un reintento (opción [2] "Pedir nuevas categorías" del menú de confirmación) si ya
    # había una propuesta previa en el estado — se detecta directamente sobre el dato, no
    # buscando texto en el historial de mensajes (frágil: dependía de que el texto exacto de
    # la opción 2 del menú no cambiara nunca).
    propuesta_previa = state.get("categorias_propuestas") or {}
    categorias_previas = propuesta_previa.get("categorias", []) if isinstance(propuesta_previa, dict) else []
    es_reintento = bool(categorias_previas)

    # Fichas técnicas completas (problema, metodología, aportaciones, limitaciones...), no solo
    # los títulos: para que la propuesta sea de verdad "de alto nivel" tiene que fundamentarse en
    # el contenido real de cada trabajo, igual que ya hace `evaluar_categorizacion_node`.
    fichas_trabajos = json.dumps(trabajos, indent=2, ensure_ascii=False)

    bloque_reintento = ""
    if es_reintento:
        resumen_previo = "\n".join(
            f"- \"{cat['nombre']}\" ({len(cat.get('trabajos', []))} trabajos): {cat['descripcion']}"
            for cat in categorias_previas
        )
        bloque_reintento = f"""
    PROPUESTA ANTERIOR (el usuario la ha rechazado y pide una propuesta nueva y mejor):
    {resumen_previo}

    Antes de proponer, evalúa CRÍTICAMENTE esa propuesta anterior a partir del recuento de trabajos
    de cada categoría: ¿hay categorías con muy pocos trabajos frente a otras sobrecargadas?
    ¿son demasiado amplias, demasiado estrechas, o se solapan entre sí? ¿reflejan bien el problema y
    la metodología real de los trabajos o son superficiales? Genera una propuesta NUEVA y REALMENTE
    DISTINTA que corrija esos problemas: no repitas los mismos nombres de categoría ni el mismo
    criterio de división (si antes fue por temática, prueba por metodología, tipo de arquitectura,
    dominio de aplicación u otro eje relevante — y viceversa).
    """

    prompt = f"""
    Eres un editor de revistas científicas. Clasifica estos trabajos para la sección 'Related Works'.

    Analiza en profundidad la ficha técnica de cada trabajo (problema específico, metodología,
    aportaciones y limitaciones) para fundamentar la categorización en su contenido real, no solo
    en el título.

    FICHAS TÉCNICAS DE LOS TRABAJOS:
    {fichas_trabajos}

    TEMA DEL PAPER DEL USUARIO: {tema}
    {bloque_reintento}
    ESTILO REQUERIDO:
    1. NOMBRE CATEGORÍA: Máximo 5 palabras. Debe ser un concepto técnico de alto nivel.
    2. NO uses frases como "Investigación sobre..." o "El trabajo de...".
    3. DESCRIPCIÓN: Una sola frase técnica y directa que describa la categoría, justificada por el contenido real de los trabajos que agrupa.
    4. En el campo "trabajos" de cada categoría, usa EXACTAMENTE el título de cada trabajo tal y como aparece en las fichas técnicas.
    """

    try:
        # Forzamos una temperatura baja para evitar nombres creativos largos
        res = llm_complejo.with_structured_output(PropuestaCategorias).invoke(prompt)
        propuesta_dict = res.model_dump()

        asignados = set()
        categorias_finales = []

        for cat in propuesta_dict['categorias']:
            nombre_limpio = cat['nombre'].strip().title()
            nombre_limpio = nombre_limpio.rstrip(".")

            validos = [t for t in cat['trabajos'] if t in titulos_reales and t not in asignados]

            if validos:
                categorias_finales.append({
                    "nombre": nombre_limpio,
                    "descripcion": cat['descripcion'],
                    "trabajos": validos
                })
                for v in validos: asignados.add(v)

        faltantes = [t for t in titulos_reales if t not in asignados]
        if faltantes and categorias_finales:
            categorias_finales[0]['trabajos'].extend(faltantes)

        cabecera = (
            "He evaluado la propuesta anterior (equilibrio entre categorías, solapamientos y "
            "profundidad de la división) y diseñado una propuesta **alternativa y mejorada**.\n"
            if es_reintento else
            "He analizado en detalle el contenido de cada trabajo (problema, metodología y "
            "aportaciones) para diseñar una propuesta de categorías de alto nivel.\n"
        )
        lineas_propuesta = [
            f"{cabecera}"
            f"A continuación se muestran las categorías propuestas:\n"
        ]

        for idx, cat in enumerate(categorias_finales, 1):
            lineas_propuesta.append(f"Categoría {idx}: **{cat['nombre']}**")
            lineas_propuesta.append(f"   *Descripción:* {cat['descripcion']}")
            lineas_propuesta.append("    *Artículos asociados:*")
            for t in cat['trabajos']:
                lineas_propuesta.append(f"      - {_truncar_texto(t, 75)}")
            lineas_propuesta.append("") # Línea en blanco de separación

        mensaje_final = "\n".join(lineas_propuesta)

        mensaje_final2 = (
            f"¿Qué deseas hacer ahora?"
            f"\n  [1] - Aceptar categorías y continuar."
            f"\n  [2] - Pedir nuevas categorías."
            f"\n  [3] - Modificar el nombre de alguna categoría."
            f"\n  [4] - Rechazar categorías y redactar de corrido."
            f"\nIntroduce el número de tu opción:"
        )

        return {
            "categorias_propuestas": {"categorias": categorias_finales},
            "error": None,
            "messages": [
                AIMessage(content=mensaje_final),
                AIMessage(content=mensaje_final2)
            ]
        }
        
    except Exception as e:
        print(f"⚠️ Error en generación taxonómica: {e}")
        return {
            "error": f"Error al proponer categorías: {str(e)}",
            "messages": [AIMessage(content=f"🚨 No se pudo consolidar la taxonomía automática: {str(e)}")]
        }

def confirmar_categorias_node(state: AgentState) -> AgentState:
    """
    Nodo: Confirmar Categorías
    Tipo: Tarea Usuario

    El usuario puede:
    1. Aceptar las categorías propuestas.
    2. Rechazarlas y pedir nuevas categorías al agente.
    3. Modificar parcialmente las categorías propuestas.
    4. Rechazar la inclusión de categorías.
    """

    # El menú de opciones ya se muestra en el AIMessage del nodo anterior.
    # interrupt() debe quedar FUERA de cualquier try/except: internamente se
    # implementa lanzando una excepción para pausar el grafo, así que un
    # `except Exception` la capturaría como si fuera un error real y se
    # saltaría la pausa por completo.
    decision = interrupt("").strip()

    if decision not in ["1", "2", "3", "4"]:
        return {
            "error": "Respuesta inválida.",
            "messages": [
                AIMessage(content="La opción seleccionada no es válida. Debes elegir 1, 2 , 3 o 4.")
            ]
        }

    if decision == "1":
        return {
            "categorias_confirmadas": True,
            "accion_categorias": "aceptar",
            "error": None,
            "messages": [
                HumanMessage(content="Opción 1: Apruebo la estructura de categorías propuesta."),
                AIMessage(content="Excelente. Estructura fijada. Procediendo a redactar la introducción de la sección Related Works...")
            ]
        }

    # OPCIÓN 2: Pedir nuevas categorías (Regenerar con otro enfoque)
    if decision == "2":
        return {
            "categorias_confirmadas": False,
            "accion_categorias": "regenerar",
            "error": None,
            "messages": [
                HumanMessage(content="Opción 2: No me convence esta agrupación, solicita generar nuevas categorías."),
                AIMessage(content="Entendido. Reorientando el análisis para ofrecerte una alternativa...")
            ]
        }

    # OPCIÓN 3: Modificar de forma personalizada
    if decision == "3":
        instrucciones = interrupt(
            "INSTRUCCIONES DE MODIFICACIÓN\n"
            "Indica qué deseas cambiar (ej: 'Cambia el nombre de la categoría 1 a Modelos de Lenguaje' "
            "o 'Mueve el paper X a la categoría 2')."
        ).strip()

        return {
            "categorias_confirmadas": False,
            "accion_categorias": "modificar",
            "instrucciones_modificacion": instrucciones,
            "error": None,
            "messages": [
                HumanMessage(content=f"Opción 3: Deseo ajustar las categorías con los siguientes cambios: '{instrucciones}'"),
                AIMessage(content="Modificaciones registradas. Ajustando el esquema de categorías según tus indicaciones...")
            ]
        }

    # OPCIÓN 4: Rechazar y avanzar en texto plano
    if decision == "4":
        return {
            "categorizar_activo": False,
            "categorias_confirmadas": False,
            "accion_categorias": "rechazar",
            "error": None,
            "messages": [
                HumanMessage(content="Opción 4: Prefiero prescindir de las categorías y redactar la sección de corrido."),
                AIMessage(content="Entendido. Desactivando categorías. Preparando la estrategia para una redacción lineal unificada...")
            ]
        }

def gateway_categorias(state: AgentState):

    accion = state.get("accion_categorias")

    if accion == "aceptar":
        return "redactar_introduccion"

    if accion == "regenerar":
        return "proponer_categorias"

    if accion == "modificar":
        return "modificar_categorias"
    
    if accion == "rechazar":
        return "redactar_introduccion"

    return "confirmar_categorias"

def modificar_categorias_node(state: AgentState) -> AgentState:

    categorias = state["categorias_propuestas"]
    instrucciones = state.get("instrucciones_modificacion", "")
    trabajos = state.get("trabajos_analizados", [])
    titulos_reales = [t["titulo"] for t in trabajos]

    if isinstance(categorias, str):
        try:
            categorias = json.loads(categorias)
        except json.JSONDecodeError:
            return {
                "error": "El JSON de categorías actuales no es válido.",
                "messages": [
                    AIMessage(content="⚠️ Error operativo interno: El esquema de categorías previo no se pudo deserializar correctamente.")
                ]
            }

    prompt = f"""
Eres un editor de revistas científicas aplicando una edición QUIRÚRGICA sobre una taxonomía de
categorías ya existente para la sección "Related Works". Tu única tarea es aplicar EXACTAMENTE los
cambios que pide el usuario, sin rediseñar la taxonomía por tu cuenta.

CATEGORÍAS ACTUALES:
{json.dumps(categorias, indent=2, ensure_ascii=False)}

TÍTULOS VÁLIDOS DE LOS TRABAJOS (usa EXACTAMENTE estos títulos, tal cual, en el campo "trabajos"):
{json.dumps(titulos_reales, indent=2, ensure_ascii=False)}

INSTRUCCIONES DE MODIFICACIÓN DADAS POR EL USUARIO:
"{instrucciones}"

REGLAS DE EDICIÓN (OBLIGATORIAS):
1. Identifica qué categoría(s) o trabajo(s) referencia la instrucción, incluso si el usuario no usa
   el nombre exacto (usa la coincidencia más cercana por significado entre las categorías/trabajos
   actuales).
2. Aplica ÚNICAMENTE el cambio pedido. Cualquier categoría que la instrucción NO mencione ni afecte
   debe devolverse EXACTAMENTE igual: mismo nombre, misma descripción y mismos trabajos, sin
   reformular texto que nadie pidió tocar.
3. Cada título de TÍTULOS VÁLIDOS debe quedar asignado a exactamente una categoría al final. No
   dejes ningún trabajo sin categoría ni lo dupliques en varias.
4. No inventes trabajos que no estén en TÍTULOS VÁLIDOS, ni inventes categorías nuevas si la
   instrucción no lo pide explícitamente.
5. Máximo 3 categorías en el resultado final, salvo que el usuario pida explícitamente más.
6. Si creas o renombras una categoría, sigue este estilo: nombre de máximo 5 palabras y concepto
   técnico de alto nivel (ej. "Agentes Autónomos", "Arquitecturas LLM"), nunca frases como
   "Investigación sobre..." o "El trabajo de..."; descripción en una sola frase técnica y directa,
   justificada por el contenido real de los trabajos que agrupa.
7. Si la instrucción es ambigua o contradictoria y no puedes aplicarla con confianza razonable,
   aplica la interpretación más conservadora (la que menos se aleje del esquema actual) en vez de
   rediseñar la taxonomía por tu cuenta.
"""

    try:
        res = llm_complejo.with_structured_output(PropuestaCategorias).invoke(prompt)
        propuesta_dict = res.model_dump()
    except Exception as e:
        print(f"⚠️ Error al modificar categorías: {e}")
        return {
            "error": f"Error al modificar categorías: {str(e)}",
            "messages": [
                AIMessage(content="⚠️ No logré interpretar las modificaciones solicitadas en un formato estructurado seguro. Por favor, intenta reformular los cambios.")
            ]
        }

    # Misma red de seguridad que proponer_categorias_node: solo se aceptan títulos reales, sin
    # duplicados entre categorías, y cualquier trabajo que se quede sin categoría (p. ej. porque el
    # LLM lo olvidó al reasignar) se añade a la primera categoría en vez de perderse en silencio.
    asignados = set()
    categorias_finales = []
    for cat in propuesta_dict["categorias"]:
        nombre_limpio = cat["nombre"].strip().title().rstrip(".")
        validos = [t for t in cat["trabajos"] if t in titulos_reales and t not in asignados]
        if validos:
            categorias_finales.append({
                "nombre": nombre_limpio,
                "descripcion": cat["descripcion"],
                "trabajos": validos
            })
            asignados.update(validos)

    faltantes = [t for t in titulos_reales if t not in asignados]
    if faltantes and categorias_finales:
        categorias_finales[0]["trabajos"].extend(faltantes)

    if not categorias_finales:
        return {
            "error": "El modelo no devolvió categorías utilizables tras la modificación.",
            "messages": [
                AIMessage(content="⚠️ No logré aplicar las modificaciones solicitadas de forma consistente. Por favor, intenta reformular los cambios.")
            ]
        }

    nuevas = {"categorias": categorias_finales}

    lineas_resultado = [
        "🛠️ **Modificaciones aplicadas con éxito.**",
        "A continuación tienes el esquema taxonómico actualizado según tus peticiones:\n"
    ]

    for idx, cat in enumerate(nuevas["categorias"], 1):
        lineas_resultado.append(f"  📦 Nueva Categoría {idx}: **{cat['nombre']}**")
        lineas_resultado.append(f"     💡 *Descripción:* {cat['descripcion']}")
        lineas_resultado.append("     📄 *Artículos en esta sección:*")
        for t in cat['trabajos']:
            lineas_resultado.append(f"        - {_truncar_texto(t, 75)}")
        lineas_resultado.append("")

    mensaje_final = "\n".join(lineas_resultado)

    mensaje_final2 = (
            f"¿Qué deseas hacer ahora?"
            f"\n  [1] - Aceptar categorías y continuar."
            f"\n  [2] - Pedir nuevas categorías."
            f"\n  [3] - Modificar el nombre de alguna categoría."
            f"\n  [4] - Rechazar categorías y redactar de corrido."
            f"\nIntroduce el número de tu opción:"
        )

    return {
        "categorias_propuestas": nuevas,
        "error": None,
        "messages": [
            AIMessage(content=mensaje_final),
            AIMessage(content=mensaje_final2)
        ]
    }

def redactar_introduccion_node(state: AgentState) -> AgentState:
    
    # 1. Recuperamos el nuevo modelo estructurado de tema_paper
    contexto_paper = state.get("tema_paper", {})
    
    # Control de seguridad: Si por algún motivo viene como string, aplicamos fallbacks
    if isinstance(contexto_paper, str):
        dominio = contexto_paper
        titulo_sistema = "El sistema propuesto"
        aportacion = "abordar los problemas identificados en el sector"
    else:
        # Extraemos los campos correspondientes a la nueva estructura común de los papers
        dominio = contexto_paper.get("problema_especifico", "este campo de estudio")
        titulo_sistema = contexto_paper.get("titulo", "El sistema propuesto")
        aportacion = contexto_paper.get("aportaciones_clave", "ofrecer una solución optimizada")

    # 2. Obtenemos las categorías de forma segura
    categorias_propuestas = state.get("categorias_propuestas") or {}
    categorias = categorias_propuestas.get("categorias", [])
    hay_categorias = state.get("categorizar_activo")

    # BIFURCACIÓN: Evaluamos si el flujo cuenta con categorías estructuradas o es secuencial
    if hay_categorias and categorias:
        # --- CASO A: SÍ HAY CATEGORÍAS ---
        nombres_cat = ", ".join([c['nombre'] for c in categorias])
        detalles_cat = ". ".join([f"La categoría '{c['nombre']}' agrupa estudios sobre {c['descripcion'].lower()}" for c in categorias])

        prompt = f"""
        Tu única tarea es escribir una introducción académica muy breve para abrir la sección "Related Works".
        Debes generar EXACTAMENTE DOS PÁRRAFOS cortos. Está TERMINANTEMENTE PROHIBIDO generar más bloques o párrafos de texto.

        REGLAS DE FORMATO CRÍTICAS:
        - NO uses listas, bullets (*), guiones, subtítulos ni enumeraciones.
        - NO cites autores ni nombres de papers externos.
        - NO hables de tus retos técnicos ni metodologías internas. Sé directo.

        INSTRUCCIONES POR PÁRRAFO:
        - PÁRRAFO 1: Debe constar únicamente de dos frases continuas en el mismo bloque:
          1. Frase 1 (Empieza exactamente así): "Esta sección revisa y describe los trabajos relacionados con {dominio}."
          2. Frase 2 (Conecta inmediatamente con tu sistema): "En este contexto, '{titulo_sistema}' tiene como objetivo contribuir mediante {aportacion}."
        - PÁRRAFO 2: Debe explicar textualmente que la literatura previa se ha organizado en las siguientes categorías: {nombres_cat}. Añade esta descripción corrida y fluida: {detalles_cat}.
        """
    else:
        # --- CASO B: NO HAY CATEGORÍAS (Estructura Secuencial/Plana) ---
        prompt = f"""
        Tu única tarea es escribir una introducción académica muy breve para abrir la sección "Related Works".
        Debes generar EXACTAMENTE DOS PÁRRAFOS cortos. Está TERMINANTEMENTE PROHIBIDO generar más bloques o párrafos de texto.

        REGLAS DE FORMATO CRÍTICAS:
        - NO uses listas, bullets (*), guiones, subtítulos ni enumeraciones.
        - NO cites autores ni nombres de papers externos.
        - NO hables de tus retos técnicos ni metodologías internas. Sé directo.

        INSTRUCCIONES POR PÁRRAFO:
        - PÁRRAFO 1: Debe constar únicamente de dos frases continuas en el mismo bloque:
          1. Frase 1 (Empieza exactamente así): "Esta sección revisa y describe los trabajos relacionados con {dominio}."
          2. Frase 2 (Conecta inmediatamente con tu sistema): "En este contexto, '{titulo_sistema}' tiene como objetivo contribuir mediante {aportacion}."
        """

    try:
        response = llm_simple.invoke(prompt)

        texto_sucio = response.content.strip()
        lineas = texto_sucio.split('\n')

        # Filtro estricto de limpieza: elimina líneas vacías accidentales y cualquier residuo Markdown
        lineas_limpias = [
            l.strip() for l in lineas 
            if l.strip() and not l.strip().startswith(('*', '-', '1.', '#'))
        ]
        
        # Unimos asegurando la separación en dos párrafos limpios
        texto_final = "\n\n".join(lineas_limpias)

        tipo_estrategia = "estructurada por subsecciones" if hay_categorias else "lineal continua"
        mensaje_salida = (
            f"Borrador de la Introducción generado con éxito(Estrategia: {tipo_estrategia}).\n"
            f"A continuación se presenta el texto académico redactado:\n\n"
            f'"{texto_final}"\n\n'
        )

        mensaje_final2 = f"Avanzando a la redacción de los trabajos relacionados..."

        return {
            "introduccion_related_works": texto_final,
            "error": None,
            "messages": [
                AIMessage(content=mensaje_salida),
                AIMessage(content=mensaje_final2),
            ]
        }
        
    except Exception as e:
        print(f"⚠️ Error redactando introducción: {e}")
        return {
            "error": f"Error en redacción de introducción: {str(e)}",
            "messages": [AIMessage(content=f"🚨 No se pudo redactar el bloque de introducción: {str(e)}")]
        }

def redactar_trabajos_relacionados_node(state: AgentState) -> AgentState:
    
    trabajos = state.get("trabajos_analizados", [])
    categorias = state.get("categorias_propuestas", {})

    hay_categorias = state.get("categorizar_activo")

    # BIFURCACIÓN: Evaluamos si el flujo cuenta con categorías estructuradas o es secuencial
    if categorias and hay_categorias:
        # --- CASO A: SÍ HAY CATEGORÍAS ---
        
        prompt = f"""
        Actúa como un transcriptor de bases de datos académicas. Tu única función es formatear información.

        INSTRUCCIONES DE FORMATO (ESTRICTAS):
        1. Escribe el NOMBRE DE LA CATEGORÍA como un título independiente.
        2. Debajo de cada categoría, redacta EXACTAMENTE UN PÁRRAFO continuo por cada paper asignado a ella.
        3. CADA PÁRRAFO debe empezar exactamente así: "El trabajo '[TÍTULO DEL PAPER]' ([AÑO]) ..."
        4. El párrafo debe integrar obligatoriamente: Problema, Metodología, Aportaciones y Limitaciones en un solo bloque de texto fluido.
        
        PROHIBICIONES:
        - NO escribas introducciones generales a la sección.
        - NO escribas introducciones ni explicaciones a las categorías.
        - NO uses listas de puntos (bullets), guiones o enumeraciones.
        - NO uses frases como "En el campo de..." o "Otro trabajo destacado es...".
        - NO repitas información fuera del párrafo del paper.
        - No escribas nada más que los títulos de las categorías y los párrafos de los trabajos redactados.

        DATOS A PROCESAR (Agrupados por categoría):
        {json.dumps(categorias, indent=2, ensure_ascii=False)}

        DATOS TÉCNICOS DE LOS PAPERS:
        {json.dumps(trabajos, indent=2, ensure_ascii=False)}

        IDIOMA: Español académico.
        """
    else:
        # --- CASO B: NO HAY CATEGORÍAS (Redacción Secuencial Plana) ---
        # Eliminamos las comprobaciones analíticas de conteo que congelan el modelo

        prompt = f"""
        Eres un investigador redactando la sección "Related Works" de un paper científico. 
        Se te dan los siguientes trabajos relacionados para que redactes un párrafo por cada uno de ellos.

        TRABAJOS ANALIZADOS:
        {json.dumps(trabajos, indent=2, ensure_ascii=False)}
        
        TAREA:
        - Deberás redactar el cuerpo de la sección "Related Works" escribiendo un párrafo por cada uno de los papers que se te han adjuntado.

        INSTRUCCIONES:
        - Cada paper debe describirse en un párrafo
        - Cada parrafo debe comenzar asi: "El trabajo..."
        - Se debe especificar el TITULO y AÑO
        - Mantener estilo académico formal

        SE DEBE INCLUIR EN CADA PAPER:
        - objetivo
        - metodología
        - contribuciones
        - ventajas
        - limitaciones

        REGLAS:
        - No incluir introducciones, conlusiones, resuemnes ni comentarios. Solo los parrafos de los papers.
        - No usar categorías
        - No listas
        - Texto continuo
        - Que no haya ninguna texto más a parte de los parrafos de los papers

        IDIOMA:
        Español académico

        Devuelve SOLO el texto.
        """

    try:
        # Invocación directa
        response = llm_complejo.invoke(prompt)
        
        # Limpieza estándar de artefactos de formato markdown que suele arrojar el LLM
        texto_redactado = response.content.replace("###", "").replace("**", "").strip()

        # Ajustamos el mensaje de log según el flujo ejecutado
        tipo_redaccion = "con estructura de categorías" if (categorias and hay_categorias) else "en formato secuencial lineal"

        mensaje_salida = (
            f"Cuerpo del Estado del Arte redactado de forma autónoma.\n"
            f"El documento se ha generado utilizando un enfoque *{tipo_redaccion}*.\n\n"
            f"Manuscrito generado:\n\n"
            f"{texto_redactado}\n"
        )

        mensaje_final = f"Avanzando hacia la redacción de la tabla comparativa..."
        
        return {
            "related_works_section": texto_redactado,
            "error": None,
            "messages": [
                AIMessage(content=mensaje_salida),
                AIMessage(content=mensaje_final)
            ]
        }
    except Exception as e:
        print(f"⚠️ Error en la redacción del estado del arte: {e}")
        return {
            "error": f"Error en la redacción del cuerpo: {e}",
            "messages": [AIMessage(content=f"🚨 No se pudo redactar la revisión de literatura: {str(e)}")]
        }

def recomendar_tabla_node(state: AgentState):

    prompt = f"""
    Ficha técnica de nuestro Paper:
    {state["tema_paper"]}

    Trabajos Relacionados Analizados:
    {json.dumps(state["trabajos_analizados"], indent=2, ensure_ascii=False)}

    Categorías Taxonómicas (si existen):
    {json.dumps(state.get("categorias_propuestas", {}), indent=2, ensure_ascii=False)}

    Evalúa si es metodológicamente recomendable incluir, al final de la sección "Related Works",
    una tabla comparativa (matriz de características) que contraste la propuesta del autor con
    la literatura analizada.

    Responde con Sí o No junto con una breve explicación del por qué. La explicación de máximo 1 párrafo de 50 palabras.
    """

    response = llm_simple.invoke(prompt)

    mensaje_salida = (
        f"¿Te recomiendo incluir una tabla comparativa?\n"
        f"{response.content}\n"
        f"¿Cuál es tu decisión? (s/n)\n"
    )

    return {
        "recomendacion_tabla": response.content.strip(),
        "error": None,
        "messages": [AIMessage(content=mensaje_salida)]
    }

def decision_tabla_node(state: AgentState):

    # El nodo anterior ya muestra la pregunta (s/n) en su AIMessage.
    # interrupt() debe quedar FUERA del try/except: internamente se implementa
    # lanzando una excepción para pausar el grafo, y un `except Exception` la
    # capturaría como si fuera un error real, saltándose la pausa por completo.
    dec = interrupt("")

    try:
        dec = dec.strip().lower()

        # Validación básica por si el usuario introduce una opción incorrecta
        if dec not in ["s", "n"]:
            mensaje_error = "⚠️ Opción no válida. Por favor, introduce 's' para generar la tabla o 'n' para omitirla."
            return {
                "error": "Respuesta inválida en tabla.",
                "messages": [
                    HumanMessage(content=f"Intento de decisión sobre tabla: '{dec}'"),
                    AIMessage(content=mensaje_error)
                ]
            }

        activa = (dec == "s")

        # Mensajes con enfoque conversacional en primera persona
        texto_humano = "Sí, por favor, genera una tabla comparativa para resumir visualmente los trabajos." if activa else "No, prefiero avanzar sin incluir una tabla comparativa en esta sección."

        texto_agente = (
            "Elección registrada. Procediendo a analizar los papers para proponer las columnas y criterios de comparación..."
            if activa else
            "Entendido. Saltaremos la fase construcción de una tabla comparativa y avanzaremos directamente hacia las conclusiones de la sección."
        )

        return {
            "tabla_comparativa_activa": activa,
            "error": None,
            "messages": [
                HumanMessage(content=texto_humano),
                AIMessage(content=texto_agente)
            ]
        }

    except Exception as e:
        print(f"⚠️ Error en la decisión de la tabla: {e}")
        return {
            "error": f"Error en decisión de tabla: {str(e)}",
            "messages": [
                AIMessage(content=f"🚨 Ocurrió un inconveniente al registrar tu decisión sobre la tabla: {str(e)}")
            ]
        }

def gateway_tabla(state: AgentState):
    return "proponer_estructura" if state["tabla_comparativa_activa"] else "redactar_conclusion"

def proponer_estructura_node(state: AgentState):

    estructura_anterior = state.get("estructura_tabla_propuesta")

    bloque_reintento = ""
    if estructura_anterior:
        columnas_anteriores = ", ".join(f'"{c}"' for c in estructura_anterior.get("columnas", []))
        justificacion_anterior = estructura_anterior.get("justificacion", "")
        bloque_reintento = f"""
PROPUESTA ANTERIOR (el usuario la ha rechazado y pide una estructura nueva y mejor):
Columnas: {columnas_anteriores}
Justificación dada en su momento: {justificacion_anterior}

Antes de proponer, evalúa CRÍTICAMENTE esa propuesta anterior a la luz de los trabajos analizados:
¿qué columnas eran poco diferenciadoras (casi todos los trabajos comparten el mismo valor, o no hay
evidencia suficiente en las fichas para rellenarlas con confianza)? ¿alguna columna binaria debería
haber sido descriptiva por perder matices relevantes al reducirla a Sí/No (o al revés, una
descriptiva que en realidad es un hecho verificable y ganaría claridad como binaria)? ¿faltaba algún
criterio relevante para este tema concreto? Diseña una estructura NUEVA que sustituya
específicamente esas columnas débiles por otras mejor fundamentadas — no te limites a cambiar
nombres o reordenar; el conjunto de columnas debe representar una perspectiva de comparación
realmente distinta y más útil que la anterior.
"""

    prompt = f"""
Eres un investigador experto diseñando la matriz de comparación (tabla comparativa de características) de la sección "Related Works" de un paper científico, al estilo de las tablas comparativas de survey papers de referencia en el área: una tabla que permite ver de un vistazo qué capacidades técnicas concretas cubre cada trabajo y en cuáles difiere del resto.

CONTEXTO:
Tema y ficha técnica de nuestro trabajo:
{state["tema_paper"]}

Trabajos analizados (problema que abordan, metodología, aportaciones, limitaciones y casos de uso):
{json.dumps(state["trabajos_analizados"], indent=2, ensure_ascii=False)}
{bloque_reintento}
TAREA:

Diseña la tabla comparativa que mejor sirva para diferenciar a ESTOS trabajos concretos. Por cada
criterio de comparación que consideres relevante, decide de forma razonada a qué tipo de columna
pertenece:

1. COLUMNA DESCRIPTIVA: propiedad cualitativa expresable en una frase corta que perdería
   información relevante si se redujera a un sí/no — p. ej. ámbito/dominio de aplicación, objetivo o
   problema que resuelve, nivel de abstracción, componentes o elementos principales que modela, tipo
   de enfoque o metodología. Elígelas específicas para el TEMA CONCRETO de estos trabajos, no una
   lista genérica de metadatos.

2. COLUMNA DE CAPACIDAD BINARIA: criterio técnico concreto y verificable que cada trabajo cumple o
   no cumple, nombrado como una capacidad afirmable (p. ej. "Modelado de Edge", "Soporte de Big
   Data", "Generación automática de código", "Evaluación empírica"), de forma que la celda se pueda
   responder inequívocamente con Sí/No. Identifícalas analizando qué capacidades técnicas concretas
   aparecen mencionadas —o notoriamente ausentes— de forma recurrente en la metodología,
   aportaciones, limitaciones o casos de uso de VARIOS de los trabajos analizados. Deben ser
   criterios reales y diferenciadores entre los trabajos (evita capacidades que cumplan todos los
   trabajos por igual o que ninguno cumpla: si un criterio no distingue a los trabajos entre sí,
   descártalo o replantéalo como columna descriptiva en vez de forzarlo a binario). El NOMBRE de cada
   columna de capacidad debe describir la capacidad en sí (nunca formularse como pregunta ni como
   etiqueta ambigua), porque ese nombre es lo único que se usará después para saber cómo rellenar
   cada celda.

NO hay una proporción fija entre columnas descriptivas y binarias: decide la mezcla que haga la
comparación más fructífera para ESTOS trabajos concretos, no una plantilla genérica. Si para este
conjunto de trabajos apenas hay capacidades verificables que realmente los distingan entre sí, usa
mayoritaria o exclusivamente columnas descriptivas; si en cambio hay varias capacidades concretas
que sí los diferencian con claridad, prioriza columnas binarias. Justifica esa elección de mezcla
explícitamente en la justificación final.

FORMATO:

{{
  "columnas": ["Título", "..."],
  "incluye_trabajo_propio": true,
  "justificacion": ""
}}

REGLAS:

- SOLO JSON
- "columnas" es una lista plana de STRINGS (solo el nombre de cada columna, p. ej. "Soporte de Big Data"). NUNCA un objeto/diccionario con el tipo u otros campos: la distinción descriptiva/binaria que has razonado arriba se refleja SOLO en cómo nombras la columna y se justifica en el campo "justificacion", no como una clave adicional en cada elemento de la lista
- NO copies literalmente los nombres de los campos del JSON de trabajos_analizados (p. ej. "Autores", "Año", "Metodología detallada") como columnas; deriva criterios de comparación propios
- 6-10 columnas en total, incluyendo "Título" (siempre la primera)
- Las columnas deben ser distintas a las de la propuesta anterior (si existe)
- justificación obligatoria (mínimo 4 líneas): explica cada columna elegida, por qué es descriptiva o binaria, y por qué es relevante para diferenciar estos trabajos concretos; si hubo propuesta anterior, explica también qué le faltaba o le sobraba y cómo la corrige esta nueva propuesta

Si repites estructura → RESPUESTA INVÁLIDA
Si no puedes → null
"""

    # 🔁 REINTENTOS AUTOMÁTICOS
    for _ in range(3):

        res = llm_complejo.invoke(prompt)
        nueva = _normalizar_columnas(extraer_json(res.content))

        if not nueva:
            continue

        if not estructura_anterior or not estructuras_similares(estructura_anterior, nueva):
            
            # --- CONSTRUCCIÓN DEL MENSAJE CONVERSACIONAL Y MENÚ (Markdown, para que gr.Chatbot
            # lo renderice de forma legible en vez de un bloque de texto plano) ---
            columnas = nueva.get("columnas", [])
            columnas_lista = [f"{i}. {c}" for i, c in enumerate(columnas, 1)]

            justificacion = (nueva.get("justificacion") or "").strip()
            justificacion_lineas = justificacion.splitlines() or ["(sin justificación proporcionada)"]
            justificacion_cita = [f"> {linea}" if linea.strip() else ">" for linea in justificacion_lineas]

            lineas_mensaje = [
                "### 📊 Propuesta de estructura para la tabla comparativa",
                "",
                f"**Columnas propuestas** ({len(columnas)} en total):",
                "",
                *columnas_lista,
                "",
                "**Justificación metodológica:**",
                "",
                *justificacion_cita,
                "",
                "---",
                "",
                "**¿Qué deseas hacer con este diseño de tabla?**",
                "",
                "[1] - Aceptar estructura y rellenar los datos automáticamente.",
                "[2] - Pedir una nueva estructura (el agente evaluará qué columnas actuales conviene cambiar).",
                "[3] - Modificar o añadir columnas de forma personalizada.",
                "[4] - Cancelar diseño de tabla y avanzar hacia la conclusión.",
            ]

            return {
                "estructura_tabla_propuesta": nueva,
                "error": None,
                "messages": [AIMessage(content="\n".join(lineas_mensaje))]
            }

    # 🚨 FALLBACK SI FALLA TODO
    mensaje_fallback = (
        "⚠️ No logré generar automáticamente una estructura suficientemente diferente a la anterior.\n"
        "👉 Te sugiero seleccionar la opción de modificación personalizada en el siguiente paso para adaptarla a tus necesidades."
    )
    return {
        "error": "Exceso de similitud en reintentos.",
        "messages": [AIMessage(content=mensaje_fallback)]
    }

def confirmar_estructura_node(state: AgentState):

    # El menú de opciones ya se muestra en el AIMessage del nodo anterior.
    opcion = interrupt("").strip()

    if opcion == "1":
        return {
            "estructura_tabla_confirmada": True,
            "accion_estructura": "aceptar",
            "error": None,
            "messages": [
                HumanMessage(content="Acepto la estructura propuesta para la tabla."),
                AIMessage(content="Estructura aprobada. Procediendo a generar la tabla con datos de los papers analizados...")
            ]
        }

    elif opcion == "2":
        return {
            "estructura_tabla_confirmada": False,
            "accion_estructura": "nueva",
            "error": None,
            "messages": [
                HumanMessage(content="Prefiero generar una propuesta de estructura nueva."),
                    AIMessage(content="🔄 Entendido. Solicitando al analista un nuevo enfoque comparativo alternativo...")
            ]
        }

    elif opcion == "3":
        instrucciones = interrupt(
            "Indica los cambios (ej. 'Quita la columna X y añade una columna para el Dataset utilizado'):"
        ).strip()

        return {
            "estructura_tabla_confirmada": False,
            "accion_estructura": "modificar",
            "instrucciones_tabla": instrucciones,
            "error": None,
            "messages": [
                HumanMessage(content=f"Deseo modificar la estructura: {instrucciones}"),
                AIMessage(content="Rediseñando el esquema de la tabla incorporando tus instrucciones de personalización...")
            ]
        }
        
    elif opcion == "4":
        return {
            "tabla_comparativa_activa": False,
            "estructura_tabla_confirmada": False,
            "accion_estructura": "rechazar",
            "error": None,
            "messages": [
                HumanMessage(content="Rechazo la inclusión de la tabla comparativa."),
                AIMessage(content="Diseño de tabla cancelado. Guardando avances y redirigiendo el flujo hacia la redacción de las conclusiones de la sección...")
            ]
        }

    else:
        return {
            "error": "Respuesta inválida en estructura de tabla.",
            "messages": [
                AIMessage(content="⚠️ Opción inválida. Por favor, introduce un número del 1 al 4.")
            ]
        }

def modificar_estructura_node(state: AgentState):

    estructura_actual = state["estructura_tabla_propuesta"]
    instrucciones = state.get("instrucciones_tabla", "")

    prompt = f"""
Eres un investigador aplicando una edición QUIRÚRGICA sobre el diseño ya acordado de la tabla
comparativa de la sección "Related Works". Tu única tarea es aplicar EXACTAMENTE los cambios que
pide el usuario sobre la estructura actual, sin rediseñarla desde cero.

ESTRUCTURA ACTUAL:
{json.dumps(estructura_actual, indent=2, ensure_ascii=False)}

INSTRUCCIONES DE MODIFICACIÓN DADAS POR EL USUARIO:
"{instrucciones}"

REGLAS DE EDICIÓN (OBLIGATORIAS):
1. Identifica qué columna(s) referencia la instrucción, incluso si el usuario no usa el nombre
   exacto (usa la coincidencia más cercana por significado entre las columnas actuales).
2. Aplica ÚNICAMENTE el cambio pedido. Cualquier columna que la instrucción NO mencione debe
   mantenerse EXACTAMENTE igual, con el mismo nombre y en el mismo orden relativo.
3. Si el usuario pide quitar una columna, elimínala y no la sustituyas por otra salvo que lo pida
   explícitamente.
4. Si el usuario pide añadir una columna nueva, decide de forma razonada si por su naturaleza debe
   ser una columna DESCRIPTIVA (propiedad cualitativa que perdería información relevante si se
   redujera a Sí/No) o una columna de CAPACIDAD BINARIA (hecho técnico verificable que cada trabajo
   cumple o no cumple, nombrada como una capacidad afirmable, nunca como una pregunta).
5. Si el usuario pide explícitamente "nuevas columnas" o "rediseñar" sin más detalle, sí puedes
   sustituir el conjunto completo por uno nuevo, manteniendo el mismo criterio descriptiva/binaria
   razonado en el punto 4 para cada columna.
6. "Título" es siempre la primera columna y nunca se elimina salvo instrucción explícita en sentido
   contrario.
7. No cambies el número total de columnas más allá de lo que la instrucción implique directamente
   (p. ej. si pide quitar una y añadir otra, el total se mantiene; si solo pide quitar, el total
   baja en consecuencia).
8. Si la instrucción es ambigua o contradictoria y no puedes aplicarla con confianza razonable,
   aplica la interpretación más conservadora (la que menos se aleje de la estructura actual) en vez
   de rediseñar la tabla por tu cuenta.

FORMATO DE SALIDA OBLIGATORIO:

{{
  "columnas": ["Título", "..."],
  "incluye_trabajo_propio": true,
  "justificacion": ""
}}

REGLAS DE FORMATO:
- SOLO JSON, sin texto ni bloques de código markdown alrededor
- "columnas" es una lista plana de STRINGS (solo el nombre de cada columna). NUNCA un objeto con el
  tipo u otros campos: la distinción descriptiva/binaria del punto 4 se refleja en el nombre y se
  explica en la justificación, no como una clave adicional
- justificación obligatoria (mínimo 3 líneas): qué cambiaste, por qué, y por qué el resto de
  columnas se mantiene igual

Si no puedes generar JSON válido → devuelve null
"""

    res = llm_complejo.invoke(prompt)
    nueva = _normalizar_columnas(extraer_json(res.content))

    if not nueva or not nueva.get("columnas"):
        return {
            "error": "El modelo no generó un JSON de estructura válido.",
            "messages": [
                AIMessage(content="⚠️ No logré interpretar los cambios solicitados en un esquema de columnas válido. Por favor, intenta reformular tus instrucciones de modificación.")
            ]
        }

    # --- CONSTRUCCIÓN DEL MENSAJE CONVERSACIONAL Y MENÚ (mismo formato Markdown que
    # proponer_estructura_node, para que gr.Chatbot lo renderice de forma consistente) ---
    columnas = nueva.get("columnas", [])
    columnas_lista = [f"{i}. {c}" for i, c in enumerate(columnas, 1)]

    justificacion = (nueva.get("justificacion") or "").strip()
    justificacion_lineas = justificacion.splitlines() or ["(sin justificación proporcionada)"]
    justificacion_cita = [f"> {linea}" if linea.strip() else ">" for linea in justificacion_lineas]

    lineas_mensaje = [
        "### 📊 Estructura de la tabla comparativa modificada",
        "",
        f"**Columnas actualizadas** ({len(columnas)} en total):",
        "",
        *columnas_lista,
        "",
        "**Justificación metodológica:**",
        "",
        *justificacion_cita,
        "",
        "---",
        "",
        "**¿Qué deseas hacer con este diseño de tabla?**",
        "",
        "[1] - Aceptar estructura y rellenar los datos automáticamente.",
        "[2] - Pedir una nueva estructura (el agente evaluará qué columnas actuales conviene cambiar).",
        "[3] - Modificar o añadir columnas de forma personalizada.",
        "[4] - Cancelar diseño de tabla y avanzar hacia la conclusión.",
    ]

    return {
        "estructura_tabla_propuesta": nueva,
        "error": None,
        "messages": [AIMessage(content="\n".join(lineas_mensaje))]
    }

def gateway_estructura(state: AgentState):

    if state["estructura_tabla_confirmada"]:
        return "generar_tabla"

    accion = state.get("accion_estructura")

    if accion == "nueva":
        return "proponer_estructura"

    elif accion == "modificar":
        return "modificar_estructura"
    
    elif accion == "rechazar":
        return "redactar_conclusion"

    return "proponer_estructura"

def generar_tabla_node(state: AgentState):

    prompt = f"""
Eres un investigador rellenando la tabla comparativa de la sección "Related Works" de un paper científico, siguiendo exactamente la estructura de columnas ya acordada.

CONTEXTO:

Trabajo Propio:
{json.dumps(state["tema_paper"], indent=2, ensure_ascii=False)}

Trabajos analizados:
{json.dumps(state["trabajos_analizados"], indent=2, ensure_ascii=False)}

Estructura de la tabla (columnas ya acordadas, en este orden):
{json.dumps(state["estructura_tabla_propuesta"], indent=2, ensure_ascii=False)}

TAREA:

Generar la tabla comparativa completa en formato Markdown: una FILA por cada trabajo analizado (y, si "incluye_trabajo_propio" es true, una fila final para el trabajo propio), con exactamente las columnas de la estructura y en ese mismo orden.

PASO PREVIO OBLIGATORIO — clasifica internamente cada columna antes de rellenar (no muestres esta clasificación en la salida):
- COLUMNA BINARIA: su nombre describe una capacidad, funcionalidad o característica afirmable que un trabajo tiene o no tiene (p. ej. "Modelado de Edge", "Soporte de Big Data", "Generación de código", "Evaluación empírica", "Código abierto").
- COLUMNA DESCRIPTIVA: el resto de columnas (ámbito, objetivo, nivel de abstracción, componentes, metodología, etc.).
- "Título" es siempre la columna de identificación: nunca se trata como binaria ni se deja vacía.

REGLAS CRÍTICAS PARA COLUMNAS BINARIAS (OBLIGATORIAS):
- El valor debe ser EXACTAMENTE "Sí" o "No" (nunca "Parcial", "No especificado", "N/A" ni ninguna explicación adicional)
- Marca "Sí" solo si hay evidencia explícita o claramente inferible en problema_especifico/metodologia_detallada/aportaciones_clave/casos_uso de que el trabajo cubre esa capacidad
- Si no hay evidencia de que el trabajo la cubra, marca "No" (la ausencia de mención se interpreta como que no la soporta); nunca dejar la celda ambigua o vacía
- Aplica el mismo criterio de evaluación por igual a todos los trabajos, incluido el trabajo propio (no lo favorezcas sistemáticamente sin evidencia)

REGLAS CRÍTICAS PARA COLUMNAS DESCRIPTIVAS (OBLIGATORIAS):
- Solo palabras clave o frases cortas, separadas por comas
- Máximo 8-12 palabras por celda
- NO escribir frases largas ni texto narrativo
- Si no hay información → escribir "No especificado" (PROHIBIDO dejar la celda vacía)

REGLAS GENERALES:
- NO incluir categorías temáticas en ninguna celda
- NO añadir columnas extra ni omitir ninguna de la estructura
- SALIDA: SOLO la tabla en Markdown (cabecera + fila separadora + filas de datos), sin texto antes o después, sin explicaciones ni comentarios

EJEMPLO DE CELDA DESCRIPTIVA CORRECTA:
"Edge computing, baja latencia, movilidad"

EJEMPLO DE CELDA DESCRIPTIVA INCORRECTA:
"Este trabajo propone una arquitectura que..."

EJEMPLO DE CELDA BINARIA CORRECTA:
"Sí"  /  "No"

EJEMPLO DE CELDA BINARIA INCORRECTA:
"Parcialmente, solo en el módulo X"

Si no puedes cumplir TODAS las reglas, la respuesta es inválida.
"""

    res = llm_complejo.invoke(prompt)

    tabla_markdown = res.content.strip()

    # --- REPORTE CONVERSACIONAL UNIFICADO ---
    mensaje_final = (
        "Tabla Comparativa Generada.\n"
        f"{tabla_markdown}\n\n"
    )

    mensaje_final2 = f"Avanzando a la descripción de la tabla..."

    return {
        "tabla_comparativa_generada": tabla_markdown,
        "error": None,
        "messages": [
            AIMessage(content=mensaje_final),
            AIMessage(content=mensaje_final2)
        ]
    }

def describir_tabla_node(state: AgentState):

    prompt = f"""
Eres un investigador redactando un paper científico.

CONTEXTO:

Tema del paper:
{state["tema_paper"]}

Estructura de la tabla:
{json.dumps(state["estructura_tabla_propuesta"], indent=2, ensure_ascii=False)}

Tabla generada:
{state["tabla_comparativa_generada"]}

TAREA:

Redactar la descripción académica de la tabla comparativa.

ESTRUCTURA OBLIGATORIA:

1. PÁRRAFO INICIAL:
- Introducir la tabla
- Explicar qué representa (comparación entre trabajos y propuesta)
- Mencionar que se basa en ciertos criterios

Ejemplo de estilo:
"In Table X, a comparison between the analyzed works and the proposed approach is presented based on the following criteria:"

2. LISTA DE CRITERIOS:
- Explicar cada columna de la tabla como un criterio de comparación
- Formato tipo lista con guiones o viñetas
- Para cada columna:
  - Nombre de la columna
  - Explicación clara de qué mide o representa

Ejemplo de estilo:
• Columna: explicación breve

REGLAS:

- NO repetir el contenido de la tabla
- NO describir cada paper
- NO inventar columnas (usar SOLO las de la estructura)
- Estilo académico formal
- Explicaciones claras y concisas
- Cada criterio debe tener 1 línea (máximo 2)

IDIOMA:
Español académico

SALIDA:
Solo el texto (sin encabezados tipo "Descripción de la tabla")
"""

    res = llm_simple.invoke(prompt)
    texto_descripcion = res.content.strip()

    # --- REPORTE CONVERSACIONAL UNIFICADO ---
    mensaje_final = (
        " Descripción de la tabla comparativa generada.\n"
        "Este texto servirá de apoyo formal en el manuscrito para introducir los criterios analizados:\n\n"
        f'"{texto_descripcion}"\n\n'
    )

    mensaje_final2 = f"Avanzando a la redacción de las conclusiones de la sección..."

    return {
        "descripcion_tabla": texto_descripcion,
        "error": None,
        "messages": [
            AIMessage(content=mensaje_final),
            AIMessage(content=mensaje_final2)]
    }

def redactar_conclusion_node(state: AgentState):

    prompt = f"""
Eres un investigador redactando la conclusión de la sección "Related Works" de un paper científico.

CONTEXTO:

Tema del paper propio:
{state["tema_paper"]}

Trabajos analizados:
{json.dumps(state["trabajos_analizados"], indent=2, ensure_ascii=False)}

Sección de trabajos relacionados:
{state.get("related_works_section", "")}

Tabla comparativa (si existe):
{state.get("tabla_comparativa_generada", "")}

TAREA:

Redactar un único párrafo de conclusión de la sección "Related Works".

OBJETIVO:

La conclusión debe posicionar claramente el trabajo propio frente al estado del arte.

CONTENIDO OBLIGATORIO:

El párrafo DEBE incluir:

1. Síntesis general del estado del arte
2. Principales fortalezas de los trabajos existentes
3. Principales limitaciones o carencias
4. Identificación clara del gap existente
5. Explicación de cómo el trabajo propio aborda ese gap
6. Si es posible, mencionar trade-offs o diferencias clave

ESTILO:

- Estilo académico formal (tipo journal)
- Redacción fluida y cohesionada
- Comparación implícita (no lista)
- Uso de conectores:
  - "Sin embargo"
  - "No obstante"
  - "En contraste"
  - "Cabe destacar que"

REGLAS:

- NO usar listas
- NO usar viñetas
- NO repetir frases de la sección anterior
- NO describir papers individuales
- NO mencionar explícitamente "este trabajo" → usar formulaciones académicas:
  - "la propuesta presentada"
  - "el enfoque propuesto"

LONGITUD:

- 6 a 10 líneas aproximadamente
- Un único párrafo

IDIOMA:

Español académico

SALIDA:

Solo el párrafo.
"""

    res = llm_simple.invoke(prompt)
    parrafo_conclusion = res.content.strip()

    # --- REPORTE CONVERSACIONAL DE CIERRE DE GENERACIÓN ---
    mensaje_final = (
        " Párrafo de Conclusión generado.\n"
        f'"{parrafo_conclusion}"\n\n'
    )

    mensaje_final2 = f"Avanzando a la última fase para revisar, homogeneizar e incluir referencias bibliográficas..."

    return {
        "conclusion_related_work": parrafo_conclusion,
        "error": None,
        "messages": [
            AIMessage(content=mensaje_final),
            AIMessage(content=mensaje_final2)
        ]
    }

def _limpiar_fences_markdown(texto: str) -> str:
    """Elimina los delimitadores de bloque de código Markdown (```latex ... ```) que el LLM
    añade a veces pese a las instrucciones; el fragmento debe poder pegarse tal cual en un
    documento LaTeX."""
    texto = re.sub(r'^\s*```(?:latex)?\s*\n', '', texto)
    texto = re.sub(r'\n\s*```\s*$', '', texto)
    return texto.strip()


def _forzar_parrafo_tras_titulos_categoria(texto: str) -> str:
    """Obliga a que cada título de categoría (\\noindent\\textbf{...}) sea su propio párrafo,
    insertando \\par antes y después. Así el primer párrafo de la categoría nunca queda pegado
    al título, sin depender de que el LLM deje líneas en blanco alrededor."""
    return re.sub(r'\\noindent\\textbf\{([^}]*)\}', r'\\par\\noindent\\textbf{\1}\\par', texto)


def _eliminar_entorno_landscape(texto: str) -> str:
    """Red de seguridad: la tabla ya no debe rotarse en `landscape` (ahora se reescala con
    `\\resizebox` en página vertical); si el LLM aun así envuelve la tabla en ese entorno pese a
    la instrucción, lo eliminamos sin tocar el contenido de la tabla."""
    for envoltorio in ("\\begin{landscape}", "\\end{landscape}"):
        texto = texto.replace(envoltorio + "\n", "").replace(envoltorio, "")
    return texto


def _eliminar_longtable(texto: str) -> str:
    """Red de seguridad: la tabla ya no debe paginarse con `longtable` (ahora es una única
    tabla reescalada con `\\resizebox`); si el LLM aun así usa `longtable` pese a la
    instrucción, lo convertimos a `tabular` normal y eliminamos los comandos de cabecera
    repetida (`\\endfirsthead`, `\\endhead`, etc.) que no son válidos dentro de `tabular`."""
    if "\\begin{longtable}" not in texto:
        return texto
    texto = texto.replace("\\begin{longtable}", "\\begin{tabular}").replace("\\end{longtable}", "\\end{tabular}")
    for comando in ("\\endfirsthead", "\\endhead", "\\endfoot", "\\endlastfoot"):
        texto = texto.replace(comando, "")
    return texto


def _forzar_wrap_columnas_tabla(texto: str) -> str:
    """Red de seguridad: una columna de tipo simple `l`/`c`/`r` (sin ancho fijo) nunca envuelve
    el texto, así que basta con que su cabecera o alguna celda sea más ancha que el hueco
    disponible para que el texto se salga visualmente sobre la columna vecina. Convertimos
    cualquier `l`/`c`/`r` suelto del spec de columnas en un `p{2cm}` con la alineación
    equivalente (`\\raggedright`/`\\centering`/`\\raggedleft`), dejando intactos los tokens que
    ya son `p{...}` o llevan un `>{...}` propio."""
    marcador = "\\begin{tabular}{"
    inicio = texto.find(marcador)
    if inicio == -1:
        return texto

    # El spec de columnas puede contener llaves anidadas (`>{\raggedright\arraybackslash}`,
    # `p{2cm}`...), así que hay que localizar su `}` de cierre contando profundidad en vez de
    # cortar en la primera `}` que aparezca (eso truncaría el spec dentro del primer `>{...}`).
    pos = inicio + len(marcador)
    profundidad = 1
    while pos < len(texto) and profundidad > 0:
        if texto[pos] == '{':
            profundidad += 1
        elif texto[pos] == '}':
            profundidad -= 1
        pos += 1
    if profundidad != 0:
        return texto  # llaves desbalanceadas: no tocamos nada

    fin_spec = pos - 1  # índice del `}` de cierre del spec
    spec_original = texto[inicio + len(marcador):fin_spec]

    alineacion_a_comando = {"l": "\\raggedright", "c": "\\centering", "r": "\\raggedleft"}

    def reemplazar_token(m):
        if m.group(1):
            return m.group(1)  # ya es >{...}/p{...}: no tocar
        return f">{{{alineacion_a_comando[m.group(2)]}\\arraybackslash}}p{{2cm}}"

    nueva_spec = re.sub(r'(>\{[^}]*\}|p\{[^}]*\})|([lcr])', reemplazar_token, spec_original)

    if nueva_spec == spec_original:
        return texto
    return texto[:inicio + len(marcador)] + nueva_spec + texto[fin_spec:]


MARCADORES_ORDEN_SECCIONES = [
    "% SECCION:INTRODUCCION",
    "% SECCION:CUERPO",
    "% SECCION:DESCRIPCION_TABLA",
    "% SECCION:TABLA",
    "% SECCION:CONCLUSION",
    "% SECCION:BIBLIOGRAFIA",
]


def _forzar_orden_secciones(texto: str) -> str:
    """Red de seguridad: el ensamblado final se genera en una única pasada de texto libre y el
    LLM a veces intercala fragmentos de un bloque dentro de otro (p. ej. deja parte de la
    descripción de la tabla después de la propia tabla). Le pedimos que marque el inicio de
    cada bloque con un comentario LaTeX exclusivo (invisible en el PDF) y aquí reconstruimos el
    documento completo recortando por esos marcadores y reordenando los bloques en el orden
    correcto, sea cual sea el orden en que el LLM los haya escrito. Si falta algún marcador (el
    LLM no siguió la instrucción), no tocamos nada y devolvemos el texto tal cual."""
    posiciones = {}
    for marcador in MARCADORES_ORDEN_SECCIONES:
        idx = texto.find(marcador)
        if idx == -1:
            return texto
        posiciones[marcador] = idx

    marcadores_por_posicion = sorted(posiciones.items(), key=lambda par: par[1])

    bloques = {}
    for i, (marcador, idx) in enumerate(marcadores_por_posicion):
        fin = marcadores_por_posicion[i + 1][1] if i + 1 < len(marcadores_por_posicion) else len(texto)
        bloques[marcador] = texto[idx + len(marcador):fin].strip("\n")

    return "\n\n".join(bloques[marcador] for marcador in MARCADORES_ORDEN_SECCIONES)


def _ajustar_anchos_columnas_segun_contenido(texto: str) -> str:
    """Red de seguridad: el LLM asigna el ancho `p{Xcm}` de cada columna "a ojo" y en columnas
    binarias tiende a guiarse por el contenido corto de las celdas de datos (`Sí`/`No`),
    ignorando que la cabecera de esa misma columna puede ser mucho más larga (p.ej. "Modelado
    de Edge"), lo que hace que la cabecera se salga de la columna e invada la columna vecina.
    Recalculamos el ancho de cada columna a partir de la palabra más larga que aparece en
    cualquier celda de esa columna (cabecera incluida), ya que una palabra sin espacios es lo
    único que un `p{}` no puede partir en varias líneas y por tanto lo único que realmente
    puede desbordarse."""
    marcador = "\\begin{tabular}{"
    inicio = texto.find(marcador)
    fin_tabular = texto.find("\\end{tabular}")
    if inicio == -1 or fin_tabular == -1 or fin_tabular < inicio:
        return texto

    # Localizamos el `}` de cierre del spec contando profundidad (puede contener llaves
    # anidadas, igual que en _forzar_wrap_columnas_tabla).
    pos = inicio + len(marcador)
    profundidad = 1
    while pos < len(texto) and profundidad > 0:
        if texto[pos] == '{':
            profundidad += 1
        elif texto[pos] == '}':
            profundidad -= 1
        pos += 1
    if profundidad != 0:
        return texto

    fin_spec = pos - 1
    spec = texto[inicio + len(marcador):fin_spec]
    cuerpo = texto[pos:fin_tabular]

    tokens = re.findall(r'>\{[^}]*\}p\{[^}]*\}|p\{[^}]*\}|[lcr|]', spec)
    columnas = [t for t in tokens if t != '|']
    if not columnas:
        return texto

    filas_celdas = []
    for fragmento in re.split(r'\\\\', cuerpo):
        fila = fragmento.replace('\\hline', '')
        if not fila.strip():
            continue
        filas_celdas.append(re.split(r'(?<!\\)&', fila))

    def texto_visible(cadena: str) -> str:
        # Aproximación: sustituye \comando{arg} por su argumento y elimina comandos sin
        # argumento, para no contar los backslashes/nombres de comando como si fueran texto.
        cadena = re.sub(r'\\[a-zA-Z]+\{([^{}]*)\}', r'\1', cadena)
        cadena = re.sub(r'\\[a-zA-Z]+', '', cadena)
        return cadena.replace('{', '').replace('}', '').strip()

    factor_cm_por_caracter = 0.19
    relleno_cm = 0.5
    ancho_minimo, ancho_maximo = 1.4, 6.0

    anchos = []
    for i in range(len(columnas)):
        palabra_mas_larga = 0
        for celdas in filas_celdas:
            if i >= len(celdas):
                continue
            for palabra in texto_visible(celdas[i]).split():
                palabra_mas_larga = max(palabra_mas_larga, len(palabra))
        ancho = palabra_mas_larga * factor_cm_por_caracter + relleno_cm
        anchos.append(max(ancho_minimo, min(ancho_maximo, ancho)))

    nueva_spec_partes = []
    idx_columna = 0
    for token in tokens:
        if token == '|':
            nueva_spec_partes.append('|')
            continue
        ancho = anchos[idx_columna]
        m = re.match(r'^(>\{[^}]*\})?p\{[^}]*\}$', token)
        if m and m.group(1):
            nueva_spec_partes.append(f"{m.group(1)}p{{{ancho:.2f}cm}}")
        elif m:
            nueva_spec_partes.append(f"p{{{ancho:.2f}cm}}")
        else:
            nueva_spec_partes.append(token)
        idx_columna += 1

    nueva_spec = ''.join(nueva_spec_partes)
    if nueva_spec == spec:
        return texto
    return texto[:inicio + len(marcador)] + nueva_spec + texto[fin_spec:]


def _espaciar_columnas_tabla(texto: str) -> str:
    """El `\\tabcolsep` por defecto de LaTeX (6pt) deja casi sin aire el texto de columnas
    contiguas en una tabla densa de varias columnas estrechas, pudiendo confundir dónde acaba
    una columna y empieza la siguiente. Lo ampliamos justo alrededor de la tabla y lo
    restauramos justo después, para no afectar a otras tablas del documento final donde se
    pegue este fragmento."""
    tabcolsep_ampliado = "\\setlength{\\tabcolsep}{8pt}\n"
    tabcolsep_normal = "\n\\setlength{\\tabcolsep}{6pt}"

    inicio = "\\begin{tabular}"
    fin = "\\end{tabular}"
    if inicio in texto and tabcolsep_ampliado not in texto:
        texto = texto.replace(inicio, tabcolsep_ampliado + inicio, 1)
        texto = texto.replace(fin, fin + tabcolsep_normal, 1)

    return texto


def _asegurar_resizebox_tabla(texto: str) -> str:
    """Red de seguridad: si el LLM olvida envolver la tabla en `\\resizebox` pese a la
    instrucción, lo forzamos por código para garantizar que la tabla completa siempre se
    reescale a `\\textwidth` y quepa en una única página vertical, sin depender de que el LLM
    lo recuerde en cada generación."""
    if "\\begin{tabular}" not in texto or "\\resizebox" in texto:
        return texto
    texto = texto.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\n\\begin{tabular}", 1)
    texto = texto.replace("\\end{tabular}", "\\end{tabular}\n}", 1)
    return texto


def revision_final_node(state: AgentState):
    
    # Datos de control y contexto
    idioma = state.get("idioma_salida","")
    paper_propio = state.get("tema_paper", [])
    trabajos = state.get("trabajos_analizados", [])
    categorias = state.get("categorias_propuestas", "")
    estructura_tabla = state.get("estructura_tabla_propuesta", "")

    # Textos reales que deben ser fusionados
    introduccion = state.get("introduccion_related_works", "")
    cuerpo = state.get("related_works_section", "")
    descripcion = state.get("descripcion_tabla", "")
    tabla = state.get("tabla_comparativa_generada", "")
    conclusion = state.get("conclusion_related_work", "")

    # 1. CONSTRUIMOS EL ENTORNO DE BIBLIOGRAFÍA EN LATEX (thebibliography)
    guias_cita_latex = []
    items_bibliografia = []
    
    for idx, t in enumerate(trabajos, 1):
        titulo = t.get("titulo", "Sin título")
        anio = t.get("anio") or t.get("year") or "s.f."
        autores = t.get("autores") or t.get("authors") or "Et al."
        
        if isinstance(autores, list) and len(autores) > 0:
            autores_str = ", ".join(autores)
        else:
            autores_str = str(autores)
            
        # Creamos la guía para que el LLM estampe la clave de citación exacta de LaTeX
        guias_cita_latex.append(f"- Para el paper '{titulo}': Insertar el comando de cita '\\cite{{ref-{idx}}}'")
        
        # Estructuramos el bibitem para el entorno final
        items_bibliografia.append(f"\\bibitem{{ref-{idx}}} {autores_str}. ``{titulo},'' {anio}.")

    guia_citas_str = "\n".join(guias_cita_latex)
    
    # Bloque final de bibliografía estructurado para LaTeX
    bloque_bibliografia_latex = (
        "\\begin{thebibliography}{" + str(len(trabajos)) + "}\n" +
        "\n".join(items_bibliografia) +
        "\n\\end{thebibliography}"
    )

    prompt = f"""
    Actúas como un editor académico senior y experto en tipografía científica LaTeX, encargado del pulido final de la sección "Related Works" de un paper científico. Tu trabajo tiene DOS FASES obligatorias: primero REVISAS y MEJORAS el contenido de los textos provistos, y después los ENSAMBLAS en un único documento LaTeX profesional. La salida debe ser EXCLUSIVAMENTE el código LaTeX final (nunca muestres las fases por separado).

    ==================================================
    FASE 1 — REVISIÓN Y MEJORA DE CONTENIDO
    ==================================================
    Antes de fusionar nada, revisa internamente estos tres textos y genera una versión mejorada de cada uno (conservando el idioma original y las ideas/datos de fondo, sin inventar información nueva sobre los trabajos):

    1. INTRODUCCIÓN: Púlela para que funcione como una introducción canónica de una sección "Related Works": debe contextualizar brevemente el ámbito, indicar el criterio de organización usado (por categorías o de forma cronológica/temática) y preparar al lector para el cuerpo que sigue. Corrige transiciones abruptas, repeticiones o frases genéricas de relleno.

    2. DESCRIPCIÓN DE LA TABLA COMPARATIVA: Revísala y mejórala en claridad y tono académico. Debe introducir la tabla y explicar sus criterios de comparación de la mejor manera posible siguiendo un formato tipo listado. La primera frase del párrafo introductorio DEBE mencionar explícitamente la tabla mediante la referencia LaTeX `Tabla~\\ref{{tab:related_works_comparativa}}` (con ese `~` y esa clave exactos) en vez de un número escrito a mano como "Tabla 1": LaTeX resolverá el número real automáticamente a partir del `\\label{{}}` que se añade a la tabla en el punto 4 de la Fase 2, sea cual sea la posición final de esta tabla dentro del paper completo donde se pegue este fragmento.

    3. CONCLUSIÓN (la revisión más importante y exhaustiva de las tres): Reescríbela para que actúe como un cierre comparativo real entre el estado del arte analizado y el trabajo propio, usando como referencia la ficha técnica del trabajo propio:
    {json.dumps(paper_propio, indent=2, ensure_ascii=False) if isinstance(paper_propio, dict) else paper_propio}
       La conclusión final DEBE:
       - Resumir de forma sintética lo visto en los trabajos relacionados (fortalezas y debilidades/inconvenientes comunes).
       - Señalar explícitamente el gap o vacío que queda sin resolver en el estado del arte.
       - Explicar cómo el trabajo propio (título, metodología y aportaciones de la ficha técnica anterior) cubre precisamente ese gap, realzando su valor frente a lo existente.
       - Mantenerse como prosa académica fluida en un único párrafo (sin listas ni viñetas), sin perder ninguna idea válida ya presente en el borrador original de la conclusión.

    ==================================================
    FASE 2 — ENSAMBLADO EN LATEX
    ==================================================
    Usa las versiones YA MEJORADAS de la Fase 1 (introducción, descripción de tabla y conclusión) junto con el resto de textos tal cual se proveen, y ensámblalas siguiendo estas reglas:

    INSTRUCCIONES DE FORMATO LATEX (CRÍTICAS):
    1. ESTRUCTURA DE SECCIONES: Utiliza el comando `\\section{{Related Works}}` al inicio. Si hay categorías, NO uses `\\subsection{{}}` ni ningún comando de sección/subsección para los títulos de categoría: "Related Works" no tiene subsecciones propias, así que estos títulos deben ser tipográficamente discretos, no divisorios. En su lugar, antes de los párrafos de cada categoría, inserta una línea con el nombre de la categoría en negrita (sin color), precedido de su número romano en mayúsculas seguido de paréntesis, con este formato exacto: `\\par\\noindent\\textbf{{I) Nombre de la Categoría}}\\par` (usa I, II, III, IV... en el orden en que aparecen las categorías, nunca números arábigos). Los `\\par` son obligatorios e inmediatamente pegados al `\\textbf{{}}` (sin depender de líneas en blanco): el título de categoría DEBE quedar en su propio párrafo, y el primer párrafo de un trabajo de esa categoría nunca debe empezar en la misma línea/párrafo que el título.
    2. CITAS EN EL TEXTO: En el cuerpo de los trabajos relacionados, busca dónde se menciona cada paper. Justo después de escribir el título de un trabajo, debes insertar su comando de cita correspondiente, con este formato exacto: "Título del trabajo \\cite{{ref-x}}". Sigue estrictamente esta guía de mapeo:
    {guia_citas_str}
    3. CITAS EN LA TABLA: En la celda del título de cada paper dentro de la tabla de LaTeX, debes incluir también su respectivo comando `\\cite{{ref-x}}` junto al título.
    4. FORMATO Y AJUSTE DE LA TABLA A LA PÁGINA (CRÍTICO — la tabla debe quedar SIEMPRE como una única tabla compacta en una página vertical normal, nunca partida en varias páginas ni cortada por el margen): Transforma la tabla comparativa actual (que viene en Markdown) a LaTeX siguiendo SIEMPRE estas reglas, sin excepción y sin evaluar el número de columnas:
       a. ORIENTACIÓN Y ESCALADO OBLIGATORIOS: Usa SIEMPRE una página vertical normal (nunca el entorno `landscape`). Envuelve el `tabular` completo dentro de `\\resizebox{{\\textwidth}}{{!}}{{ ... }}`, de modo que LaTeX reescale automáticamente toda la tabla (incluida la letra) al ancho exacto de la página, sin importar cuántas columnas o filas tenga ni cuánto texto lleve cada celda. Esto es obligatorio siempre: nunca dejes la tabla a tamaño natural sin `\\resizebox`.
       b. UNA SOLA TABLA, SIN PAGINACIÓN: NUNCA uses el entorno `longtable` ni ningún mecanismo que reparta la tabla en varias páginas. Usa siempre `\\begin{{table}}[h]` con un único `tabular` interno dentro del `\\resizebox`, con TODAS las filas (cabecera + un paper por fila + trabajo propio) juntas en esa única tabla, por muchas filas que tenga: al estar reescalada con `\\resizebox`, siempre cabe en el ancho de la página aunque el texto quede más pequeño — eso es intencionado y aceptable (se asume que el lector puede hacer zoom en el PDF si hace falta).
       c. ANCHO DE COLUMNAS Y AJUSTE DE LÍNEA (CRÍTICO para que ninguna palabra se salga de su celda e invada la columna vecina): PROHIBIDO usar los tipos de columna simples `l`, `c` o `r` sin ancho fijo, para CUALQUIER columna, incluidas las columnas binarias de Sí/No. Ese tipo de columna NUNCA envuelve el texto —ni el de la celda ni el de la cabecera—, así que basta con que la cabecera o una celda midan más que el hueco disponible para que el texto invada visualmente la columna de al lado. Usa SIEMPRE `p{{Xcm}}` para TODAS las columnas (idealmente `>{{\\raggedright\\arraybackslash}}p{{Xcm}}` para columnas de texto descriptivo, o `>{{\\centering\\arraybackslash}}p{{Xcm}}` para columnas binarias de Sí/No), donde X es un ancho en centímetros: esto obliga a LaTeX a partir el texto en varias líneas dentro de la celda —incluida la fila de cabecera— en lugar de desbordarlo. Da a las columnas binarias un ancho mínimo de 1.5cm para que su cabecera quepa cómodamente en 2-3 líneas, y reparte el resto del ancho entre las columnas descriptivas de forma proporcional a la cantidad de texto esperado en cada una (p. ej. la columna de Título puede llevar más ancho que las demás). No te preocupes por que la suma total coincida con `\\textwidth`: como la tabla completa se reescala después con `\\resizebox`, lo único relevante es la proporción relativa entre columnas, no su ancho absoluto.
       d. TAMAÑO DE FUENTE: usa el tamaño de letra normal del documento dentro de la tabla (NO uses `\\small` ni `\\footnotesize` manualmente); el escalado final para que quepa en la página ya lo controla `\\resizebox` automáticamente.
       e. CENTRADO: añade `\\centering` dentro del entorno `table`, inmediatamente antes del `\\resizebox`.
       f. El resto del formateo estándar se mantiene: `\\hline` para separar cabecera y filas, y escapar cualquier carácter conflictivo de LaTeX (%, _, &, etc.) que aparezca en el contenido de las celdas.
       g. PALABRAS COMPUESTAS CON "/": Si en una cabecera o celda aparecen dos palabras unidas por "/" sin espacios (p. ej. "Ventajas/Desventajas"), sustituye ese "/" literal por el comando `\\slash{{}}` (nunca dejes un "/" a pelo en ese caso). LaTeX trata "/" como un carácter no divisible; `\\slash{{}}` se ve igual pero permite partir la línea en ese punto.
       h. TÍTULO Y NUMERACIÓN DE LA TABLA (OBLIGATORIO): justo después de `\\centering`, añade un `\\caption{{...}}` breve y académico que resuma qué compara la tabla (basado en sus columnas y en el tema del paper), y justo después del `\\caption{{}}` añade `\\label{{tab:related_works_comparativa}}` (usa EXACTAMENTE esa clave, sin variarla, para que coincida con la referencia `\\ref{{}}` del punto 2 de la Fase 1). NUNCA escribas tú mismo un número de tabla fijo como "Tabla 1": dejando `\\caption{{}}` + `\\label{{}}` en la tabla y `\\ref{{}}` en la descripción, LaTeX calculará el número real automáticamente según la posición de esta tabla dentro del paper completo donde se pegue el fragmento.
    5. TEXTO CONTINUO: Asegúrate de que los párrafos se unifiquen sin costuras ortográficas o mayúsculas erróneas producto de la concatenación. Usa salto de línea doble en LaTeX para separar párrafos.
    6. HOMOGENEIZACIÓN: Unifica el registro y el tono de todos los bloques (introducción, cuerpo, descripción de tabla, conclusión) para que se lean como un único texto académico coherente, sin cambios bruscos de estilo entre secciones.

    ORDEN DEL DOCUMENTO LATEX (ESTRICTO — bloques consecutivos, PROHIBIDO intercalar contenido de un bloque dentro de otro; por ejemplo, nunca dejes una frase de la descripción de la tabla suelta después de la propia tabla): Para que se pueda verificar automáticamente que el orden es correcto, escribe el comentario LaTeX exacto indicado entre comillas al principio de cada bloque, en su propia línea, EXACTAMENTE UNA VEZ cada uno y en este orden (son comentarios `%`, invisibles al compilar, así que no afectan al PDF):
    1. "% SECCION:INTRODUCCION" seguido de `\\section{{Related Works}}` y el texto de la INTRODUCCIÓN (versión mejorada de la Fase 1).
    2. "% SECCION:CUERPO" seguido del CUERPO DE TRABAJOS RELACIONADOS completo (con las marcas de categoría en negrita y numeración romana si hay categorías, y comandos `\\cite{{}}` insertados).
    3. "% SECCION:DESCRIPCION_TABLA" seguido del texto ÍNTEGRO de la DESCRIPCIÓN DE LA TABLA COMPARATIVA (versión mejorada de la Fase 1): el párrafo completo debe ir aquí, sin dejar ninguna frase suelta para después de la tabla, y debe incluir la referencia `Tabla~\\ref{{tab:related_works_comparativa}}` según el punto 2 de la Fase 1.
    4. "% SECCION:TABLA" seguido del entorno completo de la TABLA (`\\begin{{table}}[h]` con `\\caption{{}}` y `\\label{{tab:related_works_comparativa}}` según el punto 4.h anterior, y el `tabular` envuelto en `\\resizebox{{\\textwidth}}{{!}}{{...}}`, según las reglas del punto 4 anterior), con citas en la columna del título.
    5. "% SECCION:CONCLUSION" seguido de la CONCLUSIÓN DE LA SECCIÓN (versión mejorada de la Fase 1).
    6. "% SECCION:BIBLIOGRAFIA" seguido del entorno de BIBLIOGRAFÍA (copia exactamente el bloque de bibliografía en LaTeX provisto abajo, sin modificarlo).

    PROHIBICIONES ABSOLUTAS:
    - NO utilices sintaxis Markdown (*, #, **, etc.) en ninguna parte del output. Todo debe ser LaTeX.
    - NO envuelvas el resultado entre delimitadores de bloque de código Markdown (``` ```latex ``` ``` , ``` ``` ```, etc.). El output debe ser LaTeX puro de principio a fin: la primera línea debe ser literalmente `\\section{{Related Works}}` y la última debe ser literalmente `\\end{{thebibliography}}`, sin ningún carácter antes ni después.
    - NO añadas preámbulos de documento completo (`\\documentclass`, `\\begin{{document}}`, etc.). Solo el fragmento del capítulo.
    - NO agregues textos de saludo, explicaciones, ni comentarios sobre las fases de revisión. El output debe empezar directamente con el comando `\\section` y ser únicamente el documento LaTeX final.

    TEXTOS BASE A REVISAR Y FUSIONAR:
    - INTRODUCCIÓN (borrador): {introduccion}
    - CUERPO DE TRABAJOS (no requiere reescritura de contenido, solo insertar citas): {cuerpo}
    - DESCRIPCIÓN DE TABLA (borrador): {descripcion}
    - TABLA COMPARATIVA (MARKDOWN ACTUAL): {tabla}
    - CONCLUSIÓN (borrador): {conclusion}

    BLOQUE DE BIBLIOGRAFÍA EN LATEX A PEGAR AL FINAL (cópialo tal cual, no lo reescribas):
    {bloque_bibliografia_latex}

    IDIOMA: El documento LaTeX debe estar en {idioma} pulido de alto nivel, registro académico.
    """

    try:
        response = llm_complejo.invoke(prompt)
        texto_final_latex = response.content.strip()

        # Post-procesado determinista: no confiamos en que el LLM cumpla siempre estas
        # reglas de formato al pie de la letra, así que las forzamos por código.
        texto_final_latex = _limpiar_fences_markdown(texto_final_latex)
        texto_final_latex = _forzar_orden_secciones(texto_final_latex)
        texto_final_latex = _forzar_parrafo_tras_titulos_categoria(texto_final_latex)
        texto_final_latex = _eliminar_entorno_landscape(texto_final_latex)
        texto_final_latex = _eliminar_longtable(texto_final_latex)
        texto_final_latex = _forzar_wrap_columnas_tabla(texto_final_latex)
        texto_final_latex = _ajustar_anchos_columnas_segun_contenido(texto_final_latex)
        texto_final_latex = _espaciar_columnas_tabla(texto_final_latex)
        texto_final_latex = _asegurar_resizebox_tabla(texto_final_latex)

        # Aseguramos el bloque de cierre por si el LLM sufriera algún truncamiento menor
        if "\\begin{thebibliography}" not in texto_final_latex:
            texto_final_latex += "\n\n" + bloque_bibliografia_latex
        
        lineas_vista = texto_final_latex.split("\n")
        previsualizacion = "\n".join(lineas_vista[:6])
        lineas_totales = len(lineas_vista)

        mensaje_final = (
            "Texto revisado correctamente y generado en formato Latex \n"
            "El agente ha homogeneizado e integrado todas las referencias, tablas dimensionales y citas cruzadas de forma nativa.\n\n"
            f"Métricas de salida: {lineas_totales} líneas de código científico consolidado.\n\n"
            f"Inicio del fragmento generado:\n"
            f"```latex\n"
            f"{previsualizacion}\n"
            f"...\n"
            f"\\end{{thebibliography}}\n"
            f"```\n"
            "El documento completo e indexado ha sido guardado de forma segura en la clave `related_works_document` del estado de tu agente LangGraph listo para su exportación final."
        )

        mensaje_final2 = (
            f"Procediendo a guardar el estado del agente y guardar el texto final en un documento con extensión .tex..."
        )

        mensaje_final3 = f"Trabajo generado correctamente. Aqui termina mi trabajo. ¡Espero haberte ayudado!"

        return {
            "related_works_document": texto_final_latex,
            "error": None,
            "messages": [
                AIMessage(content=mensaje_final),
                AIMessage(content=mensaje_final2),
                AIMessage(content=mensaje_final3)
            ]
        }

    except Exception as e:
        print(f"⚠️ Error crítico en el ensamblado final: {e}")
        fallback_latex = f"\\section{{Related Works}}\n{introduccion}\n\n{cuerpo}\n\n{descripcion}\n\n% [Tabla Fallback]\n\n{conclusion}\n\n{bloque_bibliografia_latex}"
        
        mensaje_error = (
            "⚠️ **Ocurrió una anomalía durante la optimización tipográfica avanzada de LaTeX.**\n"
            "No obstante, se ha generado un bloque estructurado de contingencia (*fallback*) combinando los módulos en texto lineal."
        )
        return {
            "related_works_document": fallback_latex,
            "error": str(e),
            "messages": [AIMessage(content=mensaje_error)]
        }

# ------------------------------------ GRAFO ------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("seleccionar_idioma", seleccionar_idioma_node)
graph.add_node("definir_tematica", definir_tematica_node)
graph.add_node("buscar_pdfs", buscar_pdfs_node)
graph.add_node("leer_texto", leer_texto_node)
graph.add_node("analizar_trabajos", analizar_trabajos_node)
graph.add_node("evaluar_categorizacion", evaluar_categorizacion_node)
graph.add_node("decision_categorizacion", decision_categorizacion_node)
graph.add_node("proponer_categorias", proponer_categorias_node)
graph.add_node("confirmar_categorias", confirmar_categorias_node)
graph.add_node("redactar_introduccion", redactar_introduccion_node)
graph.add_node("redactar_trabajos_relacionados", redactar_trabajos_relacionados_node)
graph.add_node("modificar_categorias", modificar_categorias_node)
graph.add_node("recomendar_tabla", recomendar_tabla_node)
graph.add_node("decision_tabla", decision_tabla_node)
graph.add_node("proponer_estructura", proponer_estructura_node)
graph.add_node("confirmar_estructura", confirmar_estructura_node)
graph.add_node("modificar_estructura", modificar_estructura_node)
graph.add_node("generar_tabla", generar_tabla_node)
graph.add_node("describir_tabla", describir_tabla_node)
graph.add_node("redactar_conclusion", redactar_conclusion_node)
graph.add_node("revision_final", revision_final_node)

graph.add_edge(START, "seleccionar_idioma")
graph.add_edge("seleccionar_idioma", "definir_tematica")
graph.add_edge("definir_tematica", "buscar_pdfs")
graph.add_edge("buscar_pdfs", "leer_texto")
graph.add_edge("leer_texto", "analizar_trabajos")
graph.add_edge("analizar_trabajos", "evaluar_categorizacion")
graph.add_edge("evaluar_categorizacion", "decision_categorizacion")

graph.add_conditional_edges(
    "decision_categorizacion",
    gateway_categorizacion,
    {
        "proponer_categorias": "proponer_categorias",
        "redactar_introduccion": "redactar_introduccion"
    }
)

graph.add_edge("proponer_categorias", "confirmar_categorias")

graph.add_conditional_edges(
    "confirmar_categorias",
    gateway_categorias,
    {
        "redactar_introduccion": "redactar_introduccion",
        "proponer_categorias": "proponer_categorias",
        "modificar_categorias": "modificar_categorias",
    }
)

graph.add_edge("modificar_categorias", "confirmar_categorias")
graph.add_edge("redactar_introduccion", "redactar_trabajos_relacionados")
graph.add_edge("redactar_trabajos_relacionados", "recomendar_tabla")
graph.add_edge("recomendar_tabla", "decision_tabla")

graph.add_conditional_edges(
    "decision_tabla",
    gateway_tabla,
    {
        "proponer_estructura": "proponer_estructura",
        "redactar_conclusion": "redactar_conclusion"
    }
)

graph.add_edge("proponer_estructura", "confirmar_estructura")

graph.add_conditional_edges(
    "confirmar_estructura",
    gateway_estructura,
    {
        "generar_tabla": "generar_tabla",
        "proponer_estructura": "proponer_estructura",
        "modificar_estructura": "modificar_estructura",
        "redactar_conclusion": "redactar_conclusion"
    }
)

graph.add_edge("modificar_estructura", "confirmar_estructura")

graph.add_edge("generar_tabla", "describir_tabla")
graph.add_edge("describir_tabla", "redactar_conclusion")
graph.add_edge("redactar_conclusion", "revision_final")
graph.add_edge("revision_final", END)

app = graph.compile(checkpointer=MemorySaver())

# --------------------------------------------- EJECUCION ----------------------------

MENSAJE_BIENVENIDA = """Bienvenido a AI Related Works Agent
Un agente de IA diseñado para ayudarte a redactar la sección 'Related Works' de tu paper científico con calidad y mínimo esfuerzo.

Antes de empezar:
Asegúrate de haber colocado los PDFs o documentos que quieres analizar dentro de la carpeta `trabajos_relacionados`.

Configuración Inicial:
Lo primero que debemos hacer es elegir el idioma de salida de la redacción.

Opciones de idioma:
  1 → Español Académico
  2 → Inglés Académico
"""

def crear_estado_inicial() -> AgentState:
    """Construye un AgentState nuevo para arrancar una sesión del grafo (consola o GUI)."""
    return {
        "idioma_salida": "",
        "tema_paper": "",
        "trabajos_pdf": [],
        "textos_trabajos": [],
        "trabajos_analizados": [],
        "categorias_propuestas": "",
        "introduccion_related_works": "",
        "related_works_section": "",
        "accion_categorias": "",
        "recomendacion_tabla": "",
        "tabla_comparativa_activa": "",
        "estructura_tabla_propuesta": [],
        "estructura_tabla_confirmada": "",
        "accion_estructura": "",
        "instrucciones_tabla": "",
        "tabla_comparativa_generada": "",
        "descripcion_tabla": "",
        "conclusion_related_work": "",
        "messages": [AIMessage(content=MENSAJE_BIENVENIDA)],
        "error": None
    }

if __name__ == "__main__":
    import uuid

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    entrada = crear_estado_inicial()
    num_mensajes_impresos = 0
    ultimo_chunk = {}

    # NOTA: nos apoyamos en el propio contenido del stream (clave "__interrupt__" del
    # último chunk) para saber si el grafo sigue pausado o ha llegado a END, en vez de
    # `app.get_state(config).next`: cuando un nodo lanza más de un interrupt() distinto
    # (p.ej. el menú "modificar" de categorías/tabla), ese `.next` deja de ser fiable
    # justo después del primer resume, aunque el grafo siga correctamente pausado.
    while True:
        for chunk in app.stream(entrada, config, stream_mode="values"):
            ultimo_chunk = chunk
            if "__interrupt__" in chunk:
                continue
            mensajes = chunk.get("messages", [])
            for msg in mensajes[num_mensajes_impresos:]:
                if isinstance(msg, HumanMessage):
                    print(f"\n👤 Humano:\n> {msg.content}")
                elif isinstance(msg, AIMessage):
                    print(f"\n🤖 Agente AI:\n{msg.content}")
            num_mensajes_impresos = len(mensajes)

        if "__interrupt__" not in ultimo_chunk:
            # El grafo ha llegado a END
            break

        valor_interrupcion = ultimo_chunk["__interrupt__"][0].value
        if valor_interrupcion:
            print(f"\n🤖 Agente AI:\n{valor_interrupcion}")

        respuesta_usuario = input("\n> ")
        entrada = Command(resume=respuesta_usuario)

    estado_final = ultimo_chunk

    guardar_state(estado_final, "state_guardado.json")

    guardar_documento_latex(estado_final, "related_works.tex")