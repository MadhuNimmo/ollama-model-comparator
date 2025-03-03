#!/usr/bin/env python3

import gradio as gr
import requests
import time
import threading
import json

class OllamaAPI:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        
    def get_available_models(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                return [model["name"] for model in models_data["models"]]
            else:
                return []
        except Exception as e:
            print(e)
            return []
    
    def generate(self, model, prompt, system_prompt="", temperature=0.7, max_tokens=1000, timeout=180):
        start_time = time.time()
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        if system_prompt:
            payload["system"] = system_prompt
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout)
            end_time = time.time()
            if response.status_code == 200:
                response_data = response.json()
                elapsed_time = end_time - start_time
                return {
                    "response": response_data.get("response", ""),
                    "elapsed_time": round(elapsed_time, 2),
                    "tokens": len(response_data.get("response", "").split())
                }
            else:
                return {
                    "response": f"Error: HTTP {response.status_code} - {response.text}",
                    "elapsed_time": 0,
                    "tokens": 0
                }
        except requests.exceptions.Timeout:
            return {
                "response": f"Error: Request timed out after {timeout} seconds.",
                "elapsed_time": 0,
                "tokens": 0
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "elapsed_time": 0,
                "tokens": 0
            }

def create_app():
    ollama_api = OllamaAPI()
    available_models = ollama_api.get_available_models()
    with gr.Blocks(title="Ollama Model Comparator") as app:
        gr.Markdown("# 🤖 Ollama Model Comparator")
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
                    value=available_models[1] if len(available_models) >= 2 else None
                )
                custom_models = gr.Textbox(
                    label="Add Custom Models (comma-separated)", 
                    placeholder="gemma:2b,llama3:8b..."
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
        with gr.Row():
            with gr.Column():
                model1_info = gr.Markdown("### Model 1")
                model1_stats = gr.Markdown("Time: 0.00s | Tokens: 0")
                model1_output = gr.Textbox(
                    label="Output",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True
                )
            with gr.Column():
                model2_info = gr.Markdown("### Model 2")
                model2_stats = gr.Markdown("Time: 0.00s | Tokens: 0")
                model2_output = gr.Textbox(
                    label="Output",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True
                )
        status = gr.Markdown("Ready to compare models")
        gr.HTML("""
        <style>
        .gradio-container {
            max-width: 1200px !important;
        }
        </style>
        """)
        def refresh_model_list():
            models = ollama_api.get_available_models()
            if not models:
                status_msg = "No models found. Is Ollama running? Try starting it with 'ollama serve'"
            else:
                status_msg = f"Found {len(models)} models"
            return models, models, status_msg
        def add_custom_models(custom_text):
            if not custom_text:
                return available_models, available_models, ""
            new_models = [m.strip() for m in custom_text.split(",") if m.strip()]
            if not new_models:
                return available_models, available_models, ""
            existing_models = ollama_api.get_available_models()
            all_models = existing_models.copy()
            for m in new_models:
                if m not in all_models:
                    all_models.append(m)
            return all_models, all_models, f"Added custom models: {', '.join(new_models)}"
        def clear_outputs():
            return "### Model 1", "### Model 2", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Ready to compare models"
        def compare_models(model1_name, model2_name, sys_prompt, user_prompt, temp, max_tok, timeout_sec):
            if not model1_name or not model2_name:
                return "### Model 1", "### Model 2", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Please select both models"
            if not user_prompt:
                return "### Model 1", "### Model 2", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Please enter a prompt"
            status_msg = "Generating responses... This may take some time."
            model1_result = ollama_api.generate(model1_name, user_prompt, sys_prompt, temp, max_tok, timeout_sec)
            model2_result = ollama_api.generate(model2_name, user_prompt, sys_prompt, temp, max_tok, timeout_sec)
            model1_header = f"### {model1_name}"
            model2_header = f"### {model2_name}"
            model1_status = f"Time: {model1_result['elapsed_time']:.2f}s | Tokens: {model1_result['tokens']}"
            model2_status = f"Time: {model2_result['elapsed_time']:.2f}s | Tokens: {model2_result['tokens']}"
            status_msg = "Comparison complete."
            return model1_header, model2_header, model1_status, model2_status, model1_result['response'], model2_result['response'], status_msg
        refresh_btn.click(fn=refresh_model_list, outputs=[model1, model2, status])
        custom_models.change(fn=add_custom_models, inputs=[custom_models], outputs=[model1, model2, status])
        generate_btn.click(
            fn=compare_models,
            inputs=[model1, model2, system_prompt, prompt, temperature, max_tokens, timeout],
            outputs=[model1_info, model2_info, model1_stats, model2_stats, model1_output, model2_output, status]
        )
        clear_btn.click(fn=clear_outputs, outputs=[model1_info, model2_info, model1_stats, model2_stats, model1_output, model2_output, status])
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()
