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
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatOllama(model="llama3.2:3b", temperature=0.1)
llm2 = ChatOllama(model="qwen2.5:7b", temperature=0.1)
llm3 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3, 
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

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

analisis_llm = llm.with_structured_output(PaperEstructurado)
categorias_llm = llm.with_structured_output(PropuestaCategorias)

categorias_llm3 = llm3.with_structured_output(PropuestaCategorias)

# -------------------------------- METODOS AUXILIARES -------------------------------------

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

    while True:
        
        opcion = input().strip()

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
            print("⚠️ Opción inválida. Por favor, introduce un número del 1 al 2.")

def definir_tematica_node(state: AgentState) -> AgentState:

    # Capturamos la descripción del usuario
    tema_usuario = input()
    
    prompt = f"""
    Analiza la descripción del paper que está escribiendo el usuario. Tu tarea es extraer y rellenar la ficha técnica formal de SU propia investigación utilizando el esquema estructurado. Infere los aspectos técnicos basándote en su explicación.
    
    Descripción del usuario:
    "{tema_usuario}"
    """
    
    try:
        # Forzamos al LLM a escupir la estructura idéntica a la de los papers analizados
        llm_estructurado = llm.with_structured_output(PaperEstructurado)
        res_pydantic = llm_estructurado.invoke(prompt)
        
        # Guardamos como diccionario estándar
        metadatos_dict = res_pydantic.model_dump()

        mensaje_agente = f"Temática del paper recibida correctamente. Realizando un análisis y estructuración de la información."
        mensaje_agente2 = f"Ficha técnica generada: {json.dumps(metadatos_dict, ensure_ascii=False)}.\n"
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
    carpeta = "trabajos_relacionados"
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
    else:
        # Generamos una lista visual de los ficheros encontrados para que el usuario sepa cuáles se van a leer
        lista_ficheros = "\n".join([f"   📄 - {os.path.basename(f)}" for f in archivos])
        mensaje_salida = (
            f"Exploración completada con éxito. Se han indexado {num_archivos} documentos para su lectura:\n"
            f"{lista_ficheros}\n\n"
        )

        mensaje_salida2 = f"Procediendo a la extracción de texto y metadatos..."
        
    return {
        "trabajos_pdf": archivos, 
        "messages": [
            AIMessage(content=mensaje_salida),
            AIMessage(content=mensaje_salida2)
        ]
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
    
    structured_llm = llm.with_structured_output(PaperEstructurado)
    
    for i, texto in enumerate(state["textos_trabajos"]):
        fragmento = texto[:12000]
        
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
            reporte_titulos.append(f"   🔹 [{i+1}] {res.titulo[:60]}...")
            print(f"✅ FINALIZADO: {res.titulo[:50]}...")
        except Exception as e:
            print(f"❌ Error en trabajo {i+1}: {e}")
    
    lista_papers_analizados = "\n".join(reporte_titulos)
    mensaje_salida = (
        f"📋 Extracción y estructuración de la literatura completada.\n"
        f"Se han generado fichas técnicas estandarizadas para los siguientes artículos:\n"
        f"{lista_papers_analizados}\n" 
    )

    mensaje_salida2 = (
        f"Aqui se muestra la lista técnica de los trabajos analizados:\n"
        f"{analizados}"
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

    response = llm.invoke(prompt)

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

    try:

        decision = input().strip().lower()

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
    mensajes_previos = state.get("messages", [])
    
    titulos_reales = [t["titulo"] for t in trabajos]
    es_reintento = any("solicita generar nuevas categorías" in m.content for m in mensajes_previos if isinstance(m, HumanMessage))

    prompt = f"""
    Eres un editor de revistas científicas. Clasifica estos trabajos para la sección 'Related Works'.
    
    TRABAJOS: {titulos_reales}

    TEMA: {tema}

    ESTILO REQUERIDO:
    1. NOMBRE CATEGORÍA: Máximo 5 palabras. Debe ser un concepto técnico (ej. 'Agentes Autónomos', 'Sistemas Multi-Agente', 'Arquitecturas LLM').
    2. NO uses frases como "Investigación sobre..." o "El trabajo de...".
    3. DESCRIPCIÓN: Una sola frase técnica y directa que describa la categoría.
    4. ENFOQUE: {'Busca una división por METODOLOGÍA' if es_reintento else 'Busca una división por TEMÁTICA'}.
    """

    try:
        # Forzamos una temperatura baja para evitar nombres creativos largos
        res = categorias_llm.invoke(prompt)
        propuesta_dict = res.model_dump()

        asignados = set()
        categorias_finales = []

        for cat in propuesta_dict['categorias']:
            nombre_limpio = " ".join(cat['nombre'].split()[:5]).title()
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

        enfoque_str = "METODOLÓGICO" if es_reintento else "TEMÁTICO"
        lineas_propuesta = [
            f"He diseñado una propuesta de categorías bajo un enfoque **{enfoque_str}**.\n"
            f"A continuación se muestran las categorías propuestas:\n"
        ]

        for idx, cat in enumerate(categorias_finales, 1):
            lineas_propuesta.append(f"Categoría {idx}: **{cat['nombre']}**")
            lineas_propuesta.append(f"   *Descripción:* {cat['descripcion']}")
            lineas_propuesta.append("    *Artículos asociados:*")
            for t in cat['trabajos']:
                lineas_propuesta.append(f"      - {t[:75]}...")
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

    try:

        decision = input().strip()

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
            print("\n" + "-"*50)
            print("INSTRUCCIONES DE MODIFICACIÓN")
            print("Indica qué deseas cambiar (ej: 'Cambia el nombre de la categoría 1 a Modelos de Lenguaje' o 'Mueve el paper X a la categoría 2').")
            print("-"*50)
            
            instrucciones = input().strip()

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

    except Exception as e:
        print(f"⚠️ Error en la toma de decisión: {e}")
        return {
            "error": f"Error confirmando categorías: {str(e)}",
            "messages": [
                AIMessage(content=f"Excepción en el nodo de confirmación: {str(e)}")
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
    import json

    categorias = state["categorias_propuestas"]
    instrucciones = state.get("instrucciones_modificacion", "")

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
    Eres un asistente experto en organizar trabajos académicos. 
    SÓLO debes devolver un objeto JSON válido con el mismo formato de entrada.

    Categorías actuales:
    {json.dumps(categorias, indent=2, ensure_ascii=False)}

    Instrucciones de modificación dadas por el usuario:
    "{instrucciones}"

    TAREA EXCLUSIVA:
    - Modifica la estructura actual aplicando con total precisión las instrucciones del usuario.
    - Mantén un máximo de 3 categorías en el resultado final.
    - Cada trabajo debe quedar asignado a una única categoría.
    - No inventes nuevos títulos de trabajos ni añadas categorías no solicitadas.
    - Devuelve ÚNICAMENTE el bloque JSON, sin introducciones, saludos ni bloques de código markdown.

    FORMATO REQUERIDO:
    {{
      "categorias": [
        {{
          "nombre": "Nombre categoría",
          "descripcion": "Descripción breve",
          "trabajos": ["Título trabajo 1", "Título trabajo 2"]
        }}
      ]
    }}
    """
    response = llm.invoke(prompt)

    # Función para extraer JSON del texto del LLM
    def extraer_json(texto: str):
        try:
            return json.loads(texto)
        except json.JSONDecodeError:
            # Intentar buscar JSON dentro del texto
            import re
            match = re.search(r'(\{.*\})', texto, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    return None
            return None

    nuevas = extraer_json(response.content)

    if not nuevas or "categorias" not in nuevas:
        return {
            "error": "El modelo no devolvió un formato JSON procesable.",
            "messages": [
                AIMessage(content="⚠️ No logré interpretar las modificaciones aplicadas en un formato estructurado seguro. Por favor, intenta reformular los cambios.")
            ]
        }

    lineas_resultado = [
        "🛠️ **Modificaciones aplicadas con éxito.**",
        "A continuación tienes el esquema taxonómico actualizado según tus peticiones:\n"
    ]

    for idx, cat in enumerate(nuevas.get('categorias', []), 1):
        lineas_resultado.append(f"  📦 Nueva Categoría {idx}: **{cat.get('nombre')}**")
        lineas_resultado.append(f"     💡 *Descripción:* {cat.get('descripcion')}")
        lineas_resultado.append("     📄 *Artículos en esta sección:*")
        for t in cat.get('trabajos', []):
            lineas_resultado.append(f"        - {t[:75]}...")
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
        response = llm.invoke(prompt)
        
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
        response = llm.invoke(prompt)
        
        # Limpieza estándar de artefactos de formato markdown que suele arrojar el LLM
        texto_redactado = response.content.replace("###", "").replace("**", "").strip()

        # Ajustamos el mensaje de log según el flujo ejecutado
        tipo_redaccion = "con estructura de categorías" if (categorias and hay_categorias) else "en formato secuencial lineal"

        previsualizacion = "\n".join(texto_redactado.split("\n")[:8])
        
        mensaje_salida = (
            f"Cuerpo del Estado del Arte redactado de forma autónoma.\n"
            f"El documento se ha generado utilizando un enfoque *{tipo_redaccion}*.\n\n"
            f"Previsualización del manuscrito:\n"
            f"{previsualizacion}\n"
            f"   [... El texto continúa analizando de manera fluida el resto de las obras ...]\n\n"
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
    Eres un agente experto y revisor de revistas científicas indexadas. 
    A partir de los datos provistos, debes realizar una recomendación formal sobre si es metodológicamente 
    necesario o enriquecedor incluir una tabla comparativa (matriz de características) al final de la sección 
    "Related Works" para contrastar la propuesta del autor con la literatura analizada.

    Ficha técnica de nuestro Paper:
    {state["tema_paper"]}

    Trabajos Relacionados Analizados:
    {json.dumps(state["trabajos_analizados"], indent=2, ensure_ascii=False)}

    Categorías Taxonómicas:
    {json.dumps(state.get("categorias_propuestas", {}), indent=2, ensure_ascii=False)}

    FORMATO REQUERIDO OBLIGATORIO:
    Recomendación: [Escribe SÍ o NO]
    Justificación: [Escribe una justificación científica, breve y con razones de peso]
    """

    res = llm.invoke(prompt)

    contenido_recomendacion = res.content.strip()

    # --- PARSING ESTRATÉGICO PARA DISEÑO VISUAL ---
    # Detectamos si el LLM recomienda un SÍ o un NO de manera robusta
    es_si = "recomendación: sí" in contenido_recomendacion.lower() or "recomendación: si" in contenido_recomendacion.lower()
    
    # Limpiamos las etiquetas repetitivas del texto para formatearlo nosotros de forma más bella
    justificacion_limpia = contenido_recomendacion
    if "justificación:" in contenido_recomendacion.lower():
        justificacion_limpia = contenido_recomendacion.lower().split("justificación:")[1].strip()
    elif "justificacion:" in contenido_recomendacion.lower():
         justificacion_limpia = contenido_recomendacion.lower().split("justificacion:")[1].strip()

    # --- REPORTE CONVERSACIONAL DE ALTA CALIDAD ---
    lineas_mensaje = []
    if es_si:
        lineas_mensaje.append("Recomendación de estructura: Se recomienda la inclusión de una Tabla Comparativa.")
    else:
        lineas_mensaje.append("Recomendación de estructura: No se considera crítica una Tabla Comparativa.")
        
    lineas_mensaje.append(f"\nJustificación: {justificacion_limpia.capitalize()}")

    mensaje_final = "\n".join(lineas_mensaje)

    mensaje_final2 = f"¿Deseas incluir la tabla comparativa? (s/n):"

    return {
        "recomendacion_tabla": contenido_recomendacion,
        "error": None,
        "messages": [
            AIMessage(content=mensaje_final),
            AIMessage(content=mensaje_final2)
        ]
    }

def decision_tabla_node(state: AgentState):

    try:
        # Lanzamos el input de forma clara guiando al usuario
        dec = input().strip().lower()

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

    prompt = f"""
Eres un investigador experto diseñando tablas comparativas.

Tu objetivo es generar una NUEVA estructura que aporte una perspectiva diferente.

CONTEXTO:
Tema:
{state["tema_paper"]}

Trabajos:
{json.dumps(state["trabajos_analizados"], indent=2, ensure_ascii=False)}

Estructura anterior:
{json.dumps(estructura_anterior, indent=2, ensure_ascii=False) if estructura_anterior else "Ninguna"}

TAREA:

Genera una NUEVA estructura de tabla comparativa.

OBLIGATORIO SI EXISTE ESTRUCTURA ANTERIOR:

- NO reutilizar más del 50% de las columnas anteriores
- Cambiar el enfoque de comparación

CAMBIO DE ENFOQUE (elige uno distinto al anterior):
- técnico (arquitectura, modelo, sistema)
- evaluativo (métricas, rendimiento)
- crítico (limitaciones, problemas)
- aplicación (casos de uso, dominios)
- comparativo (ventajas vs desventajas)

FORMATO:

{{
  "columnas": [],
  "incluye_trabajo_propio": true,
  "justificacion": ""
}}

REGLAS:

- SOLO JSON
- Trata de no incluir muchos campos del JSON de trabajos_analizados como columnas (salvo el campo título)
- 5-8 columnas
- incluir "Título"
- columnas deben ser diferentes a las anteriores
- justificación obligatoria (mínimo 4 líneas)

Si repites estructura → RESPUESTA INVÁLIDA
Si no puedes → null
"""

    # 🔁 REINTENTOS AUTOMÁTICOS
    for _ in range(3):

        res = llm.invoke(prompt)
        nueva = extraer_json(res.content)

        if not nueva:
            continue

        if not estructura_anterior or not estructuras_similares(estructura_anterior, nueva):
            
            # --- CONSTRUCCIÓN DEL MENSAJE CONVERSACIONAL Y MENÚ ---
            columnas_formateadas = ", ".join([f"[{c}]" for c in nueva.get("columnas", [])])
            
            lineas_mensaje = [
                "Propuesta de Estructura para la Tabla Comparativa\n",
                f" Columnas sugeridas: {columnas_formateadas}",
                f" Justificación metodológica: {nueva.get('justificacion')}\n",
                " ¿Qué deseas hacer con este diseño de tabla?",
                "  [1] - Aceptar estructura y rellenar los datos automáticamente.",
                "  [2] - Pedir una nueva estructura (forzará un enfoque analítico alternativo).",
                "  [3] - Modificar o añadir columnas de forma personalizada.",
                "  [4] - Cancelar diseño de tabla y avanzar hacia la conclusión."
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

    opcion = input().strip()

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
        instrucciones = input("\nIndica los cambios (ej. 'Quita la columna X y añade una columna para el Dataset utilizado'):\n> ").strip()
            
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
        print("⚠️ Opción inválida. Por favor, introduce un número del 1 al 4.")

def modificar_estructura_node(state: AgentState):

    estructura_actual = state["estructura_tabla_propuesta"]
    instrucciones = state.get("instrucciones_tabla", "")

    prompt = f"""
Eres un sistema que SOLO devuelve JSON válido.

Estructura actual:
{json.dumps(estructura_actual, indent=2, ensure_ascii=False)}

Instrucciones del usuario:
{instrucciones}

TAREA:
Modificar la estructura de la tabla según las instrucciones del usuario.

REGLAS:
- NO inventar columnas fuera de las instrucciones si son explícitas
- SI el usuario pide "nuevas columnas", rediseñar completamente
- Mantener entre 5 y 8 columnas
- Mantener formato correcto

FORMATO:

{{
  "columnas": [],
  "incluye_trabajo_propio": true,
  "justificacion": ""
}}

IMPORTANTE:
- Devuelve SOLO JSON
- Justificación obligatoria (mínimo 4 líneas)
- NO texto fuera del JSON

Si no puedes generar JSON válido → devuelve null
"""

    res = llm.invoke(prompt)
    nueva = extraer_json(res.content)

    if not nueva or "columnas" not in nueva:
        return {
            "error": "El modelo no generó un JSON de estructura válido.",
            "messages": [
                AIMessage(content="⚠️ No logré interpretar los cambios solicitados en un esquema de columnas válido. Por favor, intenta reformular tus instrucciones de modificación.")
            ]
        }

    # --- REPORTE VISUAL UNIFICADO EN UN ÚNICO MENSAJE ---
    columnas_actualizadas = ", ".join([f"[{c}]" for c in nueva.get("columnas", [])])
    
    lineas_mensaje = [
        " Esquema de la tabla modificado y personalizado correctamente.",
        f" Columnas finales de la matriz: {columnas_actualizadas}",
        f" Nueva justificación de diseño: {nueva.get('justificacion')}\n",
    ]

    mensaje_final = (
        f" ¿Qué deseas hacer con este diseño de tabla?",
            f"  [1] - Aceptar estructura y rellenar los datos automáticamente.",
            f"  [2] - Pedir una nueva estructura (forzará un enfoque analítico alternativo).",
            f"  [3] - Modificar o añadir columnas de forma personalizada.",
            f"  [4] - Cancelar diseño de tabla y avanzar hacia la conclusión."
    )

    return {
        "estructura_tabla_propuesta": nueva,
        "error": None,
        "messages": [
            AIMessage(content="\n".join(lineas_mensaje)),
            AIMessage(content=mensaje_final),
        ]
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
Eres un investigador redactando una tabla comparativa para un paper científico.

CONTEXTO:

Trabajo Propio:
{json.dumps(state["tema_paper"], indent=2, ensure_ascii=False)}

Trabajos analizados:
{json.dumps(state["trabajos_analizados"], indent=2, ensure_ascii=False)}

Estructura de la tabla:
{json.dumps(state["estructura_tabla_propuesta"], indent=2, ensure_ascii=False)}

TAREA:

Generar una tabla comparativa en formato Markdown con todos los papers realcionados y el trabajo propio.

FORMATO:

- Cada FILA = un paper
- Columnas según la estructura definida

REGLAS CRÍTICAS (OBLIGATORIAS):

1. TODAS las celdas deben estar rellenas
   - Si no hay información → escribir: "No especificado"
   - PROHIBIDO dejar celdas vacías

2. ESTILO DE CONTENIDO:
   - Solo palabras clave o frases cortas
   - Separadas por comas
   - Máximo 8-12 palabras por celda
   - NO escribir frases largas
   - NO texto narrativo

3. CONTENIDO:
   - NO incluir categorías en ninguna celda
   - NO añadir columnas extra
   - Respetar exactamente la estructura dada

4. SALIDA:
   - SOLO la tabla en Markdown
   - NO añadir texto antes o después
   - NO explicaciones
   - NO comentarios

EJEMPLO DE CELDA CORRECTA:
"Edge computing, baja latencia, movilidad"

EJEMPLO DE CELDA INCORRECTA:
"Este trabajo propone una arquitectura que..."

Si no puedes cumplir TODAS las reglas, la respuesta es inválida.
"""

    res = llm.invoke(prompt)

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

    res = llm.invoke(prompt)
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

    res = llm.invoke(prompt)
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
    Actúas como un transcriptor experto en tipografía científica y LaTeX. Tu única misión es fusionar los textos provistos en el apartado "TEXTOS A FUSIONAR" y devolver la sección "Related Works" estructurada EXCLUSIVAMENTE en código LaTeX profesional.

    INSTRUCCIONES DE FORMATO LATEX (CRÍTICAS):
    1. ESTRUCTURA DE SECCIONES: Utiliza el comando `\\section{{Related Works}}` al inicio. Para los subtítulos (si hay categorías), utiliza `\\subsection{{Nombre de la Categoría}}`.
    2. CITAS EN EL TEXTO: En el cuerpo de los trabajos relacionados, busca dónde se menciona cada paper. Justo después de escribir el título de un trabajo, debes insertar su comando de cita correspondiente. Sigue estrictamente esta guía de mapeo:
    {guia_citas_str}
    3. CITAS EN LA TABLA: En la celda del título de cada paper dentro de la tabla de LaTeX, debes incluir también su respectivo comando `\\cite{{ref-x}}`.
    4. TRADUCCIÓN DE LA TABLA A LATEX: Transforma la tabla comparativa actual (que viene en Markdown) a un entorno profesional de LaTeX utilizando `\\begin{{table}}[h]`, `\\centering`, y el entorno `\\begin{{tabular}}`. Utiliza `\\hline` para separar las cabeceras y las filas adecuadamente. Asegúrate de escapar caracteres conflictivos de LaTeX si aparecen en el texto (como % o _).
    5. TEXTO CONTINUO: Asegúrate de que los párrafos se unifiquen sin costuras ortográficas o mayúsculas erróneas producto de la concatenación. Usa salto de línea doble en LaTeX para separar párrafos.

    ORDEN DEL DOCUMENTO LATEX (ESTRICTO):
    1. `\\section{{Related Works}}`
    2. Texto de la INTRODUCCIÓN.
    3. CUERPO DE TRABAJOS RELACIONADOS (Con sus subsecciones si hay categorías y comandos `\\cite{{}}` insertados).
    4. DESCRIPCIÓN DE LA TABLA COMPARATIVA.
    5. Entorno completo de la TABLA (`\\begin{{table}}` ... `\\end{{table}}`).
    6. CONCLUSIÓN DE LA SECCIÓN.
    7. Entorno de BIBLIOGRAFÍA (Copia exactamente el bloque de bibliografía en LaTeX provisto abajo).

    PROHIBICIONES ABSOLUTAS:
    - NO utilices sintaxis Markdown (*, #, **, etc.) en ninguna parte del output. Todo debe ser LaTeX.
    - NO añadas preámbulos de documento completo (`\\documentclass`, `\\begin{{document}}`, etc.). Solo el fragmento del capítulo.
    - NO agregues textos de saludo, explicaciones o comentarios finales sobre el idioma. El output debe empezar directamente con el comando `\\section`.

    TEXTOS A FUSIONAR:
    - INTRODUCCIÓN: {introduccion}
    - CUERPO DE TRABAJOS: {cuerpo}
    - DESCRIPCIÓN DE TABLA: {descripcion}
    - TABLA COMPARATIVA (MARKDOWN ACTUAL): {tabla}
    - CONCLUSIÓN: {conclusion}

    BLOQUE DE BIBLIOGRAFÍA EN LATEX A PEGAR AL FINAL:
    {bloque_bibliografia_latex}

    IDIOMA: {idioma} pulido de alto nivel.
    """

    try:
        response = llm.invoke(prompt)
        texto_final_latex = response.content.strip()
        
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

app = graph.compile()

# --------------------------------------------- EJECUCION ----------------------------

if __name__ == "__main__":

    mensaje_bienvenida = """Bienvenido a AI Related Works Agent
Un agente de IA diseñado para ayudarte a redactar la sección 'Related Works' de tu paper científico con calidad y mínimo esfuerzo.

Antes de empezar:
Asegúrate de haber colocado los PDFs o documentos que quieres analizar dentro de la carpeta `trabajos_relacionados`.

Configuración Inicial:
Lo primero que debemos hacer es elegir el idioma de salida de la redacción.

Opciones de idioma:
  1 → Español Académico
  2 → Inglés Académico
"""

    init_state = {
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
        "messages": [AIMessage(content=mensaje_bienvenida)],
        "error": None
    }

    print(f"\n🤖 Agente AI:\n{init_state['messages'][0].content}")

    for step in app.stream(init_state):
        for node_name, node_state in step.items():
            init_state.update(node_state)
            
            if "messages" in node_state:
                # Iteramos por todos los mensajes nuevos que ha generado este nodo
                for msg in node_state["messages"]:
                    if isinstance(msg, HumanMessage):
                        print(f"\n👤 Humano:\n> {msg.content}")
                    elif isinstance(msg, AIMessage):
                        print(f"\n🤖 Agente AI:\n{msg.content}")
                #print("\n" + "="*60) # Línea divisoria elegante al terminar el nodo

    guardar_state(init_state, "state_guardado.json")

    guardar_documento_latex(init_state, "related_works.tex")

    #print("\n--- 📊 RESUMEN FINAL ---")
    #print(json.dumps(init_state.get("trabajos_analizados", []), indent=2, ensure_ascii=False))

    #print("\n📘 INTRODUCCIÓN GENERADA:\n")
    #print(init_state["introduccion_related_works"])

    #print("\n📚 CUERPO RELATED WORKS:\n")
    #print(init_state["related_works_section"])

    #print("\n📊 TABLA:\n", init_state.get("tabla_comparativa_generada", ""))
    #print("\n🧾 DESCRIPCIÓN:\n", init_state.get("descripcion_tabla", ""))
    #print("\n📌 CONCLUSIÓN:\n", init_state.get("conclusion_related_work", ""))
    #print("\n🧾 TEXTO FINAL:\n", init_state.get("related_works_document", ""))