import os
import json
import re

from typing import Dict, TypedDict, List, Union, Annotated, Sequence, Optional
from pypdf import PdfReader
from collections import defaultdict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

llm = ChatOllama(model="llama3.2:3b", temperature=0.1)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

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

# ------------------------------------ NODOS --------------------------------------------------

def definir_tematica_node(state: AgentState):
    print("\n--- 📝 PASO 1: TEMÁTICA ---")
    tema = input("¿Sobre qué trata tu paper? (Ej: IA en medicina):\n> ")
    return {
        "tema_paper": tema,
        "messages": [HumanMessage(content=f"Tema: {tema}")]
    }

def buscar_pdfs_node(state: AgentState):
    print("\n--- 📂 PASO 2: BUSCANDO PDFs ---")
    carpeta = "trabajos_relacionados"
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
        return {"error": f"Crea la carpeta '{carpeta}' y añade los PDFs.", "messages": [AIMessage(content="Carpeta creada, pero vacía.")]}
    
    archivos = [os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.endswith(".pdf")]
    print(f"Encontrados: {len(archivos)} archivos.")
    return {"trabajos_pdf": archivos, "messages": [AIMessage(content=f"Encontrados {len(archivos)} PDFs.")]}

def leer_texto_node(state: AgentState):
    print("\n--- 📖 PASO 3: LEYENDO CONTENIDO ---")
    textos = []
    for ruta in state["trabajos_pdf"]:
        try:
            reader = PdfReader(ruta)
            contenido = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            textos.append(contenido.strip())
        except Exception as e:
            print(f"Error leyendo {ruta}: {e}")
    return {"textos_trabajos": textos}

def analizar_trabajos_node(state: AgentState):
    print(f"\n--- 🤖 ANALIZANDO {len(state['textos_trabajos'])} TRABAJOS ---")
    analizados = []
    
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
            
            data = res.dict()
            if "falta de memoria a largo plazo" in data["problema_especifico"].lower():
                 print(f"⚠️ Aviso: Posible sesgo en el problema del trabajo {i+1}")
            
            analizados.append(data)
            print(f"✅ FINALIZADO: {res.titulo[:50]}...")
        except Exception as e:
            print(f"❌ Error en trabajo {i+1}: {e}")
            
    return {"trabajos_analizados": analizados}

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

    Responde SOLO:

    Recomendación: Sí o No
    Justificación: breve explicación
    """

    response = llm.invoke(prompt)

    return {
        "messages": [AIMessage(content=response.content)]
    }
    
def decision_categorizacion_node(state: AgentState) -> AgentState:
    """
    Nodo: Decisión de categorización
    Tipo: Tarea Usuario
    Descripción: El usuario decide si categorizar o no los trabajos en base a la recomendación
    que ha realizado el agente tras el análisis de los trabajos.
    """

    try:

        print("\n👤 ¿Deseas categorizar los trabajos relacionados?")
        decision = input("Responder (s/n):\n> ").strip().lower()

        if decision not in ["s", "n"]:
            return {
                "error": "Debes responder 's' o 'n'.",
                "messages": [
                    AIMessage(content="La respuesta proporcionada no es válida. Debes responder 's' (sí) o 'n' (no).")
                ]
            }

        categorizar = decision == "s"

        return {
            "categorizar_activo": categorizar,
            "error": None,
            "messages": [
                HumanMessage(
                    content=f"El usuario ha decidido {'categorizar' if categorizar else 'no categorizar'} los trabajos relacionados."
                ),
                AIMessage(
                    content=f"Decisión registrada. {'Se procederá a categorizar los trabajos.' if categorizar else 'Se continuará sin realizar categorización.'}"
                )
            ]
        }

    except Exception as e:
        return {
            "error": f"Error en decisión de categorización: {str(e)}",
            "messages": [
                AIMessage(content=f"Se produjo un error durante la toma de decisión sobre la categorización: {str(e)}")
            ]
        }

def gateway_categorizacion(state: AgentState) -> str:
    if state.get("categorizar_activo"):
        return "proponer_categorias"
    else:
        return "redactar_introduccion"

def proponer_categorias_node(state: AgentState) -> AgentState:
    print("\n--- 🤖 PULIENDO PROPUESTA DE CATEGORÍAS ---")
    trabajos = state.get("trabajos_analizados", [])
    tema = state.get("tema_paper", "")
    mensajes_previos = state.get("messages", [])
    
    titulos_reales = [t["titulo"] for t in trabajos]
    es_reintento = any("solicita generar nuevas categorías" in m.content for m in mensajes_previos if isinstance(m, HumanMessage))

    prompt = f"""
    Eres un editor de revistas científicas. Clasifica estos trabajos para la sección 'Related Works'.
    
    TRABAJOS: {titulos_reales}

    ESTILO REQUERIDO:
    1. NOMBRE CATEGORÍA: Máximo 4 palabras. Debe ser un concepto técnico (ej. 'Agentes Autónomos', 'Sistemas Multi-Agente', 'Arquitecturas LLM').
    2. NO uses frases como "Investigación sobre..." o "El trabajo de...".
    3. DESCRIPCIÓN: Una sola frase técnica y directa.
    4. ENFOQUE: {'Busca una división por METODOLOGÍA' if es_reintento else 'Busca una división por TEMÁTICA'}.
    """

    try:
        # Forzamos una temperatura baja para evitar nombres creativos largos
        res = categorias_llm.invoke(prompt)
        propuesta_dict = res.dict()

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

        return {
            "categorias_propuestas": {"categorias": categorias_finales},
            "messages": [AIMessage(content="Propuesta pulida y estilizada generada.")]
        }
    except Exception as e:
        return {"error": str(e)}

def confirmar_categorias_node(state: AgentState) -> AgentState:
    """
    Nodo: Confirmar Categorías
    Tipo: Tarea Usuario

    El usuario puede:
    1. Aceptar las categorías propuestas.
    2. Rechazarlas y pedir nuevas categorías al agente.
    3. Modificar parcialmente las categorías propuestas.
    """

    try:

        print("\n🤖 Categorías propuestas por el agente:\n")
        print(state["categorias_propuestas"])

        print("\n👤 ¿Qué deseas hacer?")
        print("1 - Aceptar categorías")
        print("2 - Pedir nuevas categorías")
        print("3 - Modificar alguna categoría")

        decision = input("\nSelecciona una opción (1/2/3):\n> ").strip()

        if decision not in ["1", "2", "3"]:
            return {
                "error": "Respuesta inválida.",
                "messages": [
                    AIMessage(content="La opción seleccionada no es válida. Debes elegir 1, 2 o 3.")
                ]
            }

        if decision == "1":

            return {
                "categorias_confirmadas": True,
                "accion_categorias": "aceptar",
                "error": None,
                "messages": [
                    HumanMessage(content="El usuario ha aceptado las categorías propuestas por el agente."),
                    AIMessage(content="Categorías confirmadas. Se continuará con el análisis utilizando esta estructura.")
                ]
            }

        if decision == "2":

            return {
                "categorias_confirmadas": False,
                "accion_categorias": "regenerar",
                "error": None,
                "messages": [
                    HumanMessage(content="El usuario ha rechazado las categorías propuestas y solicita generar nuevas categorías."),
                    AIMessage(content="Entendido. Procederé a generar una nueva propuesta de categorización.")
                ]
            }

        if decision == "3":

            print("\nDescribe qué categoría deseas modificar o qué cambio quieres realizar.")
            print("Ejemplo: cambiar el nombre de la categoría 2 o mover el trabajo 3 a la categoría 1.\n")

            instrucciones = input("✏️ Instrucciones de modificación:\n> ")

            return {
                "categorias_confirmadas": False,
                "accion_categorias": "modificar",
                "instrucciones_modificacion": instrucciones,
                "error": None,
                "messages": [
                    HumanMessage(content=f"El usuario desea modificar la propuesta de categorías con las siguientes instrucciones: {instrucciones}"),
                    AIMessage(content="He registrado las instrucciones de modificación. Procederé a ajustar la propuesta de categorías.")
                ]
            }

    except Exception as e:
        return {
            "error": f"Error confirmando categorías: {str(e)}",
            "messages": [
                AIMessage(content=f"Se produjo un error durante la confirmación de categorías: {str(e)}")
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
                    AIMessage(content="⚠️ Error: las categorías actuales no son un JSON válido.")
                ]
            }

    prompt = f"""
Eres un asistente experto en organizar trabajos académicos. 
SÓLO devuelves JSON válido con el mismo formato de entrada.

Categorías actuales:
{json.dumps(categorias, indent=2, ensure_ascii=False)}

Instrucciones del usuario (pueden incluir):
- Eliminar una categoría por nombre.
- Mover trabajos de una categoría a otra.
- Eliminar trabajos específicos de una categoría.
- Renombrar categorías.

TAREA:
- Modifica las categorías según las instrucciones.
- Mantén máximo 3 categorías.
- Cada trabajo debe estar en una única categoría.
- No inventes nuevos trabajos ni categorías.
- Devuelve SOLO JSON, sin explicaciones.

FORMATO DE RESPUESTA:
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

    # Agregar las instrucciones del usuario al prompt
    prompt += f"\nInstrucciones del usuario:\n{instrucciones}"

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
            "error": "El modelo no devolvió JSON válido.",
            "messages": [
                AIMessage(content="⚠️ Error al modificar categorías. El modelo no devolvió un JSON válido. Inténtalo de nuevo.")
            ]
        }

    return {
        "categorias_propuestas": nuevas,
        "messages": [
            AIMessage(content="Categorías modificadas correctamente."),
            AIMessage(content=json.dumps(nuevas, indent=2, ensure_ascii=False))
        ]
    }

def redactar_introduccion_node(state: AgentState) -> AgentState:
    print("\n--- ✍️ REDACTANDO INTRODUCCIÓN (FORMATO ESTRICTO 2 PÁRRAFOS) ---")
    tema = state.get("tema_paper", "")
    categorias = state.get("categorias_propuestas", {}).get("categorias", [])
    
    nombres_cat = ", ".join([c['nombre'] for c in categorias])
    detalles_cat = ". ".join([f"La categoría '{c['nombre']}' agrupa estudios sobre {c['descripcion'].lower()}" for c in categorias])

    prompt = f"""
    Eres un redactor académico de élite. Tu objetivo es escribir EXACTAMENTE dos párrafos de texto continuo.

    REGLAS DE ORO:
    1. PROHIBIDO usar listas, bullets (*), guiones o títulos internos.
    2. PROHIBIDO citar nombres de papers o autores.
    3. PÁRRAFO 1: Debe comenzar explicando que la sección Related Works explora el estado del arte y trabajos previos relevantes para contextualizar el desarrollo de agentes LLM en la redacción científica.
    4. PÁRRAFO 2: Debe explicar que se ha optado por organizar la literatura en las categorías de {nombres_cat}. Justifica que esta división permite diferenciar entre enfoques (ej: individuales vs. colaborativos). {detalles_cat}.
    5. No añadas nada más. Ni introducciones de "Aquí tienes", ni conclusiones.

    CONTEXTO: {tema}
    """

    try:
        response = llm.invoke(prompt)
        
        texto_sucio = response.content.strip()
        
        lineas = texto_sucio.split('\n')

        lineas_limpias = [
            l for l in lineas 
            if l.strip() and not l.strip().startswith(('*', '-', '1.', '#'))
        ]
        
        texto_final = "\n\n".join(lineas_limpias)

        return {
            "introduccion_related_works": texto_final,
            "messages": [AIMessage(content="Introducción de 2 párrafos generada.")]
        }
    except Exception as e:
        return {"error": str(e)}

def redactar_trabajos_relacionados_node(state: AgentState) -> AgentState:
    print("\n--- ✍️ REDACTANDO CUERPO (SOLO CATEGORÍAS Y PAPERS) ---")
    trabajos = state.get("trabajos_analizados", [])
    categorias_json = state.get("categorias_propuestas", {}).get("categorias", [])

    if not categorias_json:
        return {"error": "No hay categorías para redactar."}

    trabajos_map = {t["titulo"]: t for t in trabajos}

    prompt = f"""
    Actúa como un transcriptor de bases de datos académicas. Tu única función es formatear información.

    INSTRUCCIONES DE FORMATO (ESTRICTAS):
    1. Escribe el NOMBRE DE LA CATEGORÍA como un título.
    2. Debajo de cada categoría, redacta UN PÁRRAFO por cada paper asignado a ella.
    3. CADA PÁRRAFO debe empezar exactamente así: "El trabajo '[TÍTULO DEL PAPER]' ([AÑO]) ..."
    4. El párrafo debe integrar: Problema, Metodología, Aportaciones y Limitaciones en un solo bloque de texto fluido.
    
    PROHIBICIONES:
    - NO escribas introducciones generales a la sección.
    - NO escribas introducciones a las categorías.
    - NO uses listas de puntos (bullets).
    - NO uses frases como "En el campo de..." o "Otro trabajo destacado es...".
    - NO repitas información fuera del párrafo del paper.

    DATOS A PROCESAR (Agrupados por categoría):
    {json.dumps(categorias_json, indent=2, ensure_ascii=False)}

    DATOS TÉCNICOS DE LOS PAPERS:
    {json.dumps(trabajos, indent=2, ensure_ascii=False)}

    IDIOMA: Español académico.
    """

    try:
        response = llm.invoke(prompt)
        
        texto_redactado = response.content.replace("###", "").replace("**", "").strip()

        return {
            "related_works_section": texto_redactado,
            "messages": [AIMessage(content="Cuerpo de Related Works generado sin introducciones.")]
        }
    except Exception as e:
        return {"error": f"Error en redacción: {e}"}

def recomendar_tabla_node(state: AgentState):

    prompt = f"""
    Eres un agente que esta redactando la sección "Related Works" de un paper de investigación.
    Con los datos que se te encomiendan a continuación, debes realizar un recomendación sobre si
    incluir o no una tabla comparativa a modo de resumen visual para comparar el paper de investigación
    con los trabajos relacionados que se exploran en esta sección que se esta redactando.

Tema:
{state["tema_paper"]}

Trabajos:
{json.dumps(state["trabajos_analizados"], indent=2, ensure_ascii=False)}

Categorías:
{json.dumps(state.get("categorias_propuestas", {}), indent=2, ensure_ascii=False)}

¿Recomiendas tabla comparativa?

Formato:
Recomendación: SI/NO
Justificación: ...

REGLAS:
- La justificación debe ser breve y debe inlcuir razones de peso sobre la inclusión o no de la tabla.
"""

    res = llm.invoke(prompt)

    return {
        "recomendacion_tabla": res.content,
        "messages": [
            AIMessage(content="Evaluando necesidad de tabla..."),
            AIMessage(content=res.content)
        ]
    }

def decision_tabla_node(state: AgentState):

    #print("\n🤖 Recomendación:\n")
    #print(state["recomendacion_tabla"])

    dec = input("\n👤 ¿Añadir tabla? (s/n): ").lower()

    return {
        "tabla_comparativa_activa": dec == "s",
        "messages": [
            HumanMessage(content=f"Usuario decide tabla: {dec}")
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

        if not estructura_anterior:
            return {
                "estructura_tabla_propuesta": nueva,
                "messages": [AIMessage(content=json.dumps(nueva, indent=2, ensure_ascii=False))]
            }

        # 🔥 VALIDAR DIFERENCIA REAL
        if not estructuras_similares(estructura_anterior, nueva):
            return {
                "estructura_tabla_propuesta": nueva,
                "messages": [
                    AIMessage(content="Nueva estructura generada correctamente"),
                    AIMessage(content=json.dumps(nueva, indent=2, ensure_ascii=False))
                ]
            }

    # 🚨 FALLBACK SI FALLA TODO
    return {
        "messages": [
            AIMessage(content="⚠️ No se pudo generar una estructura suficientemente diferente. Intenta modificar manualmente.")
        ]
    }

def confirmar_estructura_node(state: AgentState):

    print("\n📊 Estructura actual:\n")
    print(json.dumps(state["estructura_tabla_propuesta"], indent=2, ensure_ascii=False))

    print("\nOpciones:")
    print("1 → Aceptar estructura")
    print("2 → Generar nueva propuesta")
    print("3 → Modificar estructura")

    opcion = input("\n👤 Elige opción (1/2/3): ").strip()

    if opcion == "1":
        return {
            "estructura_tabla_confirmada": True,
            "accion_estructura": "aceptar",
            "messages": [HumanMessage(content="Aceptar estructura")]
        }

    elif opcion == "2":
        return {
            "estructura_tabla_confirmada": False,
            "accion_estructura": "nueva",
            "messages": [HumanMessage(content="Generar nueva estructura")]
        }

    elif opcion == "3":
        instrucciones = input(
            "\n✏️ Indica cómo modificar la estructura:\n> "
        )

        return {
            "estructura_tabla_confirmada": False,
            "accion_estructura": "modificar",
            "instrucciones_tabla": instrucciones,
            "messages": [
                HumanMessage(content=f"Modificar: {instrucciones}")
            ]
        }

    else:
        print("Opción inválida.")
        return confirmar_estructura_node(state)

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

    if not nueva:
        return {
            "messages": [
                AIMessage(content="⚠️ Error modificando estructura. Intenta de nuevo.")
            ]
        }

    return {
        "estructura_tabla_propuesta": nueva,
        "messages": [
            AIMessage(content="Estructura modificada correctamente"),
            AIMessage(content=json.dumps(nueva, indent=2, ensure_ascii=False))
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

    return "proponer_estructura"

def generar_tabla_node(state: AgentState):

    prompt = f"""
Eres un investigador redactando una tabla comparativa para un paper científico.

CONTEXTO:

Tema del paper:
{state["tema_paper"]}

Trabajos analizados:
{json.dumps(state["trabajos_analizados"], indent=2, ensure_ascii=False)}

Estructura de la tabla:
{json.dumps(state["estructura_tabla_propuesta"], indent=2, ensure_ascii=False)}

TAREA:

Generar una tabla comparativa en formato Markdown.

FORMATO:

- Cada FILA = un paper
- Columnas según la estructura definida
- Incluir una fila final: **Trabajo propio**

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

    return {
        "tabla_comparativa_generada": res.content.strip(),
        "messages": [
            AIMessage(content="Tabla comparativa generada."),
            AIMessage(content=res.content)
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

    print("\nDEBUG DESCRIPCIÓN:", repr(res.content))

    return {
        "descripcion_tabla": res.content.strip(),
        "messages": [
            AIMessage(content="Descripción de la tabla generada."),
            AIMessage(content=res.content)
        ]
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

    return {
        "conclusion_related_work": res.content.strip(),
        "messages": [
            AIMessage(content="Conclusión generada."),
            AIMessage(content=res.content)
        ]
    }

# ------------------------------------ GRAFO ------------------------------------------

graph = StateGraph(AgentState)

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

graph.add_edge(START, "definir_tematica")
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
        "modificar_estructura": "modificar_estructura"
    }
)

graph.add_edge("modificar_estructura", "confirmar_estructura")

graph.add_edge("generar_tabla", "describir_tabla")
graph.add_edge("describir_tabla", "redactar_conclusion")
graph.add_edge("redactar_conclusion", END)

app = graph.compile()

# --------------------------------------------- EJECUCION ----------------------------

if __name__ == "__main__":
    init_state = {
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
        "messages": [],
        "error": None
    }

    for step in app.stream(init_state):
        for node_name, node_state in step.items():
            init_state.update(node_state)
            
            if "messages" in node_state:
                ultimo_msg = node_state["messages"][-1]
                if isinstance(ultimo_msg, AIMessage):
                    print(f"\n🤖 Agente [{node_name}]:\n{ultimo_msg.content}")

    guardar_state(init_state, "state_guardado.json")

    print("\n--- 📊 RESUMEN FINAL ---")
    print(json.dumps(init_state.get("trabajos_analizados", []), indent=2, ensure_ascii=False))

    print("\n📘 INTRODUCCIÓN GENERADA:\n")
    print(init_state["introduccion_related_works"])

    print("\n📚 CUERPO RELATED WORKS:\n")
    print(init_state["related_works_section"])

    print("\n📊 TABLA:\n", init_state.get("tabla_comparativa_generada", ""))
    print("\n🧾 DESCRIPCIÓN:\n", init_state.get("descripcion_tabla", ""))
    print("\n📌 CONCLUSIÓN:\n", init_state.get("conclusion_related_work", ""))
