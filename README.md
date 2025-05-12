# 🏗️ Asesor Virtual para Obreros de Obra (******PROTOTIPO******)

Este proyecto tiene como objetivo crear un **chatbot** que asesore a los **obreros** en una obra de construcción. El chatbot proporcionará información útil sobre procedimientos de construcción, seguridad, herramientas y otros recursos, mejorando la eficiencia y la seguridad en el sitio de trabajo.

## 📚 Librerías utilizadas

Este proyecto hace uso de las siguientes librerías para procesar información y crear un asistente virtual interactivo:

- **Chroma**: Para almacenar y buscar documentos vectorizados, lo que permite que el chatbot recupere información relevante de manera eficiente.
- **LangChain**: Para gestionar la interacción entre el chatbot y las fuentes de datos, integrando los modelos de OpenAI con un sistema de recuperación de información.
- **OpenAI**: Para generar respuestas a partir de un modelo de lenguaje como GPT-3 o GPT-4, con la finalidad de proporcionar respuestas útiles a los obreros.
- **Gradio**: Para crear una interfaz de usuario sencilla que permita a los obreros interactuar con el chatbot en tiempo real.

## ⚙️ Instalación de dependencias

Luego, ejecuta los siguientes comandos en Git Bash o tu terminal preferida para instalar las dependencias necesarias:

```bash
pip install langchain
pip install chromadb
pip install openai
pip install gradio
