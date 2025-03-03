#!/usr/bin/env python3
"""
Ollama Model Comparator - Compare responses from different Ollama models
"""

import gradio as gr
import requests
import time
import json
from typing import List

class OllamaAPI:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        
    def get_available_models(self):
        """Get list of available models from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                return [model["name"] for model in models_data["models"]]
            else:
                return []
        except Exception as e:
            print(f"Error getting models: {e}")
            return []
    
    def generate_streaming(
        self, 
        model: str, 
        prompt: str, 
        system_prompt: str = "", 
        temperature: float = 0.7, 
        max_tokens: int = 1000, 
        timeout: int = 180
    ):
        """
        Generate a response from a specific model with streaming, 
        returning partial chunks. This function is a generator 
        yielding (text_so_far, token_count, elapsed_time).
        """
        start_time = time.time()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response_text = ""
        token_count = 0
        
        try:
            with requests.post(
                f"{self.base_url}/api/generate", 
                json=payload, 
                stream=True, 
                timeout=timeout
            ) as response:
                if response.status_code != 200:
                    error_text = f"Error: HTTP {response.status_code} - {response.text}"
                    yield error_text, 0, time.time() - start_time
                    return
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        line_decoded = line.decode("utf-8", errors="ignore")
                        if not line_decoded.startswith("{"):
                            continue
                        
                        chunk = json.loads(line_decoded)
                        
                        if "response" in chunk:
                            token = chunk["response"]
                            response_text += token
                            token_count += 1
                            elapsed = time.time() - start_time
                            yield response_text, token_count, elapsed
                        
                        if chunk.get("done", False):
                            return
                    
                    except Exception as e:
                        print(f"Error parsing chunk: {e}")
                        continue
                        
        except requests.exceptions.Timeout:
            err = (f"Error: Request timed out after {timeout} seconds. "
                   f"The model ({model}) might be too large or unavailable.")
            yield err, token_count, time.time() - start_time
        except Exception as e:
            err = f"Error: {str(e)}"
            yield err, token_count, time.time() - start_time


def get_theme():
    return gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
        text_size=gr.themes.sizes.text_md,
    )

def create_app():
    """Create and configure the Gradio app (sequential streaming)."""
    ollama_api = OllamaAPI()
    available_models = ollama_api.get_available_models()
    
    with gr.Blocks(title="Ollama Model Comparator", theme=get_theme()) as app:
        
        gr.Markdown("# 🤖 Ollama Model Comparator")
        
        # Inputs
        with gr.Row():
            with gr.Column():
                model1 = gr.Dropdown(
                    label="Model 1",
                    choices=available_models,
                    value=available_models[0] if available_models else None
                )
                model2 = gr.Dropdown(
                    label="Model 2",
                    choices=available_models,
                    value=(
                        available_models[1] 
                        if len(available_models) >= 2 
                        else (available_models[0] if available_models else None)
                    )
                )
                
                custom_models = gr.Textbox(
                    label="Add Custom Models (comma-separated)", 
                    placeholder="gemma:2b,llama3:8b...",
                    info="Enter model names that aren't in the dropdown list above"
                )
                
                refresh_btn = gr.Button("Refresh Model List")
                
            with gr.Column():
                system_prompt = gr.Textbox(
                    label="System Prompt (optional)", 
                    placeholder="You are a helpful assistant...",
                    lines=2
                )
                prompt = gr.Textbox(
                    label="Prompt", 
                    placeholder="Enter your prompt here...", 
                    lines=5
                )
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.1, value=0.7, 
                        label="Temperature"
                    )
                    max_tokens = gr.Slider(
                        minimum=100, maximum=4000, step=100, value=1000, 
                        label="Max Tokens"
                    )
                    timeout = gr.Slider(
                        minimum=30, maximum=600, step=30, value=180, 
                        label="Timeout (seconds)"
                    )
                
                with gr.Row():
                    generate_btn = gr.Button("Compare Models", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear Results")
        
        # Side-by-side comparison area
        with gr.Row(equal_height=True):
            with gr.Column():
                model1_info = gr.Markdown("### Model 1")
                model1_stats = gr.Markdown("Time: 0.00s | Tokens: 0")
                model1_output = gr.Markdown(elem_classes=["model-output"])
            
            with gr.Column():
                model2_info = gr.Markdown("### Model 2")
                model2_stats = gr.Markdown("Time: 0.00s | Tokens: 0")
                model2_output = gr.Markdown(elem_classes=["model-output"])
        
        status = gr.Markdown("Ready to compare models")
        
        with gr.Accordion("Options and Help", open=False):
            gr.Markdown("""
            ### Instructions
            1. Select two models to compare (or add custom model names)
            2. Enter your prompt (and optional system prompt)
            3. Adjust temperature, max tokens and timeout if needed
            4. Click "Compare Models" to see results side by side
            
            ### Notes
            - Models must be installed in Ollama before they can be compared
            - If you don't see your models, click "Refresh Model List"
            - For large models, you may need to increase the timeout
            - Responses are streamed in real-time as they're generated
            """)

        # --- Helper Functions ---
        def update_model_list():
            models = ollama_api.get_available_models()
            return {
                model1: gr.Dropdown.update(choices=models, value=models[0] if models else None),
                model2: gr.Dropdown.update(
                    choices=models, 
                    value=(
                        models[1] if len(models) >= 2
                        else (models[0] if models else None)
                    )
                ),
                status: f"Found {len(models)} models"
            }
        
        def update_custom_models(m1, m2, custom_text):
            """
            If the user typed custom models, append them 
            to the available list. 
            """
            if not custom_text.strip():
                # No changes if empty
                return {model1: m1, model2: m2}
            
            custom_model_list = [
                t.strip() for t in custom_text.split(",") if t.strip()
            ]
            existing = ollama_api.get_available_models()
            
            # Combine
            all_models = list({*existing, *custom_model_list})
            return {
                model1: gr.Dropdown.update(choices=all_models, value=m1),
                model2: gr.Dropdown.update(choices=all_models, value=m2),
                status: f"Added custom models: {', '.join(custom_model_list)}"
            }
        
        def clear_outputs():
            """Reset everything."""
            return {
                model1_info: "### Model 1",
                model2_info: "### Model 2",
                model1_stats: "Time: 0.00s | Tokens: 0",
                model2_stats: "Time: 0.00s | Tokens: 0",
                model1_output: "",
                model2_output: "",
                status: "Ready to compare models",
            }
        
        def compare_models(m1, m2, sys_prompt, user_prompt, temp, tokens, t_out):
            """Sequentially stream from Model 1 then Model 2, yielding partial updates."""
            if not m1 or not m2:
                yield {
                    status: "Please select both models before comparing."
                }
                return
            
            # Clear & set initial labels
            yield {
                model1_info: f"### {m1}",
                model2_info: f"### {m2}",
                model1_stats: "Time: 0.00s | Tokens: 0",
                model2_stats: "Time: 0.00s | Tokens: 0",
                model1_output: "",
                model2_output: "",
                status: f"Comparing {m1} vs {m2}..."
            }
            
            # --- Stream from Model 1 ---
            text_so_far = ""
            token_count = 0
            start_time = time.time()
            for partial_text, partial_tokens, elapsed in ollama_api.generate_streaming(
                model=m1,
                prompt=user_prompt,
                system_prompt=sys_prompt,
                temperature=temp,
                max_tokens=tokens,
                timeout=t_out
            ):
                text_so_far = partial_text
                token_count = partial_tokens
                yield {
                    model1_output: text_so_far,
                    model1_stats: f"Time: {elapsed:.2f}s | Tokens: {token_count}",
                    status: f"Generating from {m1}..."
                }
            
            # Done Model 1
            yield {
                status: f"Finished {m1}. Now generating with {m2}..."
            }
            
            # --- Stream from Model 2 ---
            text_so_far = ""
            token_count = 0
            start_time = time.time()
            for partial_text, partial_tokens, elapsed in ollama_api.generate_streaming(
                model=m2,
                prompt=user_prompt,
                system_prompt=sys_prompt,
                temperature=temp,
                max_tokens=tokens,
                timeout=t_out
            ):
                text_so_far = partial_text
                token_count = partial_tokens
                yield {
                    model2_output: text_so_far,
                    model2_stats: f"Time: {elapsed:.2f}s | Tokens: {token_count}",
                    status: f"Generating from {m2}..."
                }
            
            # Done
            yield {
                status: "Completed generating responses."
            }

        # --- Hook up events ---
        refresh_btn.click(
            fn=update_model_list,
            outputs=[model1, model2, status]
        )
        
        custom_models.change(
            fn=update_custom_models,
            inputs=[model1, model2, custom_models],
            outputs=[model1, model2, status]
        )
        
        generate_btn.click(
            fn=compare_models,
            inputs=[model1, model2, system_prompt, prompt, temperature, max_tokens, timeout],
            outputs=[model1_info, model2_info, model1_stats, model2_stats,
                     model1_output, model2_output, status],
            api_name="compare_models"
        )
        
        clear_btn.click(
            fn=clear_outputs,
            outputs=[model1_info, model2_info, model1_stats, model2_stats,
                     model1_output, model2_output, status]
        )
        
        # Inject custom CSS
        app.load(
            None,
            js="""
            () => {
                const style = document.createElement('style');
                style.textContent = `
                    .model-output {
                        min-height: 400px;
                        border: 1px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 16px;
                        overflow-y: auto;
                        background-color: #f9f9f9;
                        color: #000 !important; /* force visible text color */
                    }
                `;
                document.head.appendChild(style);
            }
            """
        )
    return app

if __name__ == "__main__":
    # For a public shareable link, pass share=True:
    #   app.launch(share=True)
    app = create_app()
    app.launch()
