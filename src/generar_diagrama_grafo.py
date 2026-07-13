"""
Genera una imagen del grafo de LangGraph definido en app1.py, para poder
incluirla en la memoria del TFG.

Uso:
    python src/generar_diagrama_grafo.py

Genera `grafo_agente.png` en la raíz del repositorio. `draw_mermaid_png()`
renderiza el diagrama a través de la API pública de mermaid.ink, así que
requiere conexión a internet; si falla (por ejemplo, sin conexión), se guarda
como alternativa el texto Mermaid en `grafo_agente.mmd`, que se puede pegar
en https://mermaid.live o compilar de forma local con mermaid-cli.
"""

import os

from app1 import app

_RAIZ_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_SALIDA_PNG = os.path.join(_RAIZ_REPO, "grafo_agente.png")
RUTA_SALIDA_MMD = os.path.join(_RAIZ_REPO, "grafo_agente.mmd")


def main():
    grafo = app.get_graph()
    try:
        png = grafo.draw_mermaid_png()
        with open(RUTA_SALIDA_PNG, "wb") as f:
            f.write(png)
        print(f"Imagen del grafo guardada en: {RUTA_SALIDA_PNG}")
    except Exception as e:
        print(f"No se pudo generar el PNG ({e}). Se guarda el texto Mermaid como alternativa.")
        with open(RUTA_SALIDA_MMD, "w", encoding="utf-8") as f:
            f.write(grafo.draw_mermaid())
        print(f"Texto Mermaid guardado en: {RUTA_SALIDA_MMD} (pégalo en https://mermaid.live para verlo).")


if __name__ == "__main__":
    main()
