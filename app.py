"""Main entry point for the app.

This app is generated based on your prompt in Vertex AI Studio using
Google GenAI Python SDK (https://googleapis.github.io/python-genai/) and
Gradio (https://www.gradio.app/).

You can customize the app by editing the code in Cloud Run source code editor.
You can also update the prompt in Vertex AI Studio and redeploy it.
"""

import base64
from google import genai
from google.genai import types
import gradio as gr
import utils


def generate(
    message,
    history: list[gr.ChatMessage],
    request: gr.Request
):
  """Function to call the model based on the request."""

  validate_key_result = utils.validate_key(request)
  if validate_key_result is not None:
    yield validate_key_result
    return

  client = genai.Client(
      vertexai=True,
      project="amorei-465607",
      location="global",
  )
  text1 = types.Part.from_text(text=f"""Provide advice and recommendations on the following topics:

1.  **Heating Systems**:
    *   Types of heating systems (gas, oil, electric, heat pumps, solar thermal).
    *   Pros and cons of each system.
    *   Suitability for different types of buildings (single-family homes, apartment buildings, commercial properties).
2.  **Energy Efficiency**:
    *   Methods to improve energy efficiency in heating.
    *   Insulation options and their impact on heating costs.
    *   Smart thermostats and energy management systems.
3.  **Regulatory Compliance**:
    *   German regulations and standards for heating systems (e.g., Heizungsgesetz, Energieeinsparverordnung - EnEV).
    *   Requirements for inspections and maintenance.
    *   Available subsidies and incentives for energy-efficient heating systems.

**Example:**

**User Input:** \"Ich möchte mein altes Heizsystem austauschen. Was sind die besten Optionen für ein energieeffizientes System in meinem Einfamilienhaus?\" (I want to replace my old heating system. What are the best options for an energy-efficient system in my single-family home?)

**Assistant Output:** \"Für ein Einfamilienhaus gibt es mehrere energieeffiziente Optionen. Eine Wärmepumpe ist sehr effizient, benötigt aber möglicherweise eine Fußbodenheizung oder Heizkörper mit großer Oberfläche. Eine Gasbrennwertheizung ist eine weitere gute Option, besonders wenn Sie bereits einen Gasanschluss haben. Solarthermie kann ebenfalls eine sinnvolle Ergänzung sein, um die Heizung zu unterstützen und Warmwasser zu bereiten. Es ist wichtig, die spezifischen Gegebenheiten Ihres Hauses (Isolierung, Heizlast etc.) zu berücksichtigen, um die beste Wahl zu treffen. Zudem sollten Sie die aktuellen Förderprogramme des BAFA und der KfW prüfen, um finanzielle Unterstützung zu erhalten.\" (For a single-family home, there are several energy-efficient options. A heat pump is very efficient but may require underfloor heating or radiators with a large surface area. A gas condensing boiler is another good option, especially if you already have a gas connection. Solar thermal energy can also be a useful addition to support heating and provide hot water. It is important to consider the specific conditions of your house (insulation, heating load, etc.) to make the best choice. You should also check the current funding programs of BAFA and KfW to receive financial support.)

Ensure your advice is practical, considers the German context, and includes references to relevant regulations and incentives where appropriate.""")
  si_text1 = types.Part.from_text(text=f"""You are a professional heating consultant in Germany, providing expert advice on heating systems, energy efficiency, and regulatory compliance. Your responses should be accurate, informative, and tailored to the German market.""")

  model = "gemini-2.5-flash"
  contents = [
      types.Content(
          role="user",
          parts=[
            text1
          ]
      )
  ]
  for prev_msg in history:
    role = "user" if prev_msg["role"] == "user" else "model"
    parts = utils.get_parts_from_message(prev_msg["content"])
    if parts:
      contents.append(types.Content(role=role, parts=parts))

  if message:
    contents.append(
        types.Content(role="user", parts=utils.get_parts_from_message(message))
    )

  tools = [
      types.Tool(google_search=types.GoogleSearch()),
  ]

  generate_content_config = types.GenerateContentConfig(
      temperature=1,
      top_p=1,
      seed=0,
      max_output_tokens=65535,
      safety_settings=[
          types.SafetySetting(
              category="HARM_CATEGORY_HATE_SPEECH",
              threshold="OFF"
          ),
          types.SafetySetting(
              category="HARM_CATEGORY_DANGEROUS_CONTENT",
              threshold="OFF"
          ),
          types.SafetySetting(
              category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
              threshold="OFF"
          ),
          types.SafetySetting(
              category="HARM_CATEGORY_HARASSMENT",
              threshold="OFF"
          )
      ],
      tools=tools,
      system_instruction=[si_text1],
  )

  results = []
  for chunk in client.models.generate_content_stream(
      model=model,
      contents=contents,
      config=generate_content_config,
  ):
    if chunk.candidates and chunk.candidates[0] and chunk.candidates[0].content:
      results.extend(
          utils.convert_content_to_gr_type(chunk.candidates[0].content)
      )
      if results:
        yield results
gr.set_static_paths(paths=["static/images/"])
with gr.Blocks(theme='shivi/calm_seafoam',title="KI Heizungsoptimierung") as demo:
    gr.HTML('<img src="/static/images/logo_amorei.jpg" alt="Amorei Logo" style="height:60px;">')
    gr.HTML("<h1><strong>A</strong>more<strong>I</strong> - ihr KI-gestützter Heizungsberater</h1>")
    with gr.Row():  # Korrekte Einrückung auf gleicher Ebene wie die gr.HTML Aufrufe
        with gr.Column(scale=1):
            with gr.Row():
                gr.HTML("<h2>Willkommen in der Heizungsberater KI Demo!</h2>")
            with gr.Row():
                gr.HTML("""Die KI Heizungsoptimierung ist ein KI-gestützter virtueller Heizungsberater, der Ihnen hilft, die besten Heizungsoptionen für Ihr Zuhause zu finden. Sie können Fragen zu Heizsystemen, Energieeffizienz und Fördermöglichkeiten stellen. Die KI wird Ihnen basierend auf Ihren Eingaben maßgeschneiderte Empfehlungen geben.
                Sie können die KI auch nach den besten Heizsystemen für Ihr Zuhause fragen, wie z.B. Wärmepumpen, Gasbrennwertheizungen oder Solarthermie. Die KI wird Ihnen auch Informationen zu den aktuellen Förderprogrammen des BAFA und der KfW geben, um Ihnen bei der Finanzierung Ihrer Heizungsoptimierung zu helfen.
                <br><br>
                <strong>Beispiel:</strong> Fragen Sie die KI nach den besten Heizsystemen für Ihr Zuhause, wie z.B. Wärmepumpen, Gasbrennwertheizungen oder Solarthermie. Die KI wird Ihnen auch Informationen zu den aktuellen Förderprogrammen des BAFA und der KfW geben, um Ihnen bei der Finanzierung Ihrer Heizungsoptimierung zu helfen.
                <br><br>
                Nutzen Sie auch die Möglichkeit, Dateien wie Bilder oder PDFs hochzuladen, um Ihre Fragen zu verdeutlichen. Die KI wird diese Dateien analysieren und Ihnen basierend auf den Inhalten der Dateien maßgeschneiderte Empfehlungen geben.""")
        with gr.Column(scale=2, variant="panel"):
            gr.ChatInterface(
                fn=generate,
                title="KI Heizungsoptimierung",
                type="messages",
                editable=True,
                multimodal=True,
                examples=[
                "Ich möchte mein altes Heizsystem austauschen. Was sind die besten Optionen für ein energieeffizientes System in meinem Einfamilienhaus?", 
                "Kann ich eine Solaranlange auf meiner 40 qm Terasse im ersten Stock eines Mietshauses installieren?",
                "Ich möchte mein elektrisch betriebenes Auto über Nacht laden mit ökologischem Strom. Was benötige ich dafür und wird das gefördert?",
                "Welche Förderungen gibt es für die Installation einer Wärmepumpe in meinem Einfamilienhaus?",
                "Wie kann ich den Energieverbrauch meiner Heizungsanlage optimieren?",
                "Welche Vorteile bietet eine Solaranlage eigentlich für mein Einfamilienhaus?"
                ],
                cache_examples=False # True gibt Fehler aus wegen Verzeichnisberechtigungen
            )
    with gr.Row():
        gr.HTML("""
        <div style="background-color: #fffacd; border: 1px solid #eedc82; padding: 20px; margin: 20px; border-radius: 5px; color: #8b4513; font-weight: bold; text-align: center;">
    <img src="/static/images/logo_amorei.jpg" alt="Amorei Logo" style="height:60px; vertical-align:middle; margin-right:15px;">
    <span style="margin-right: 10px;">⚠️</span>
    Achtung: Diese App ist nur für Demozwecke bestimmt und unterstützt noch keine User Access Control. Bitte vermeiden Sie die Verwendung sensibler Daten.
  </div>""")

demo.launch(show_error=True)
