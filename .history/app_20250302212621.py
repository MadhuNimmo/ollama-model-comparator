#!/usr/bin/env python3
"""
Ollama Model Comparator - Compare responses from different Ollama models
"""

import gradio as gr
import requests
import time
import threading
import json
from typing import Dict, List

class OllamaAPI:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        
    def get_available_models(self):
        """Get list of available models from Ollama"""
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
    
    def generate_streaming(self, model, prompt, system_prompt="", temperature=0.7, max_tokens=1000, 
                         update_callback=None, timeout=180):
        """Generate a response from a specific model with streaming updates"""
        start_time = time.time()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True  # Enable streaming
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response_text = ""
        token_count = 0
        
        try:
            with requests.post(f"{self.base_url}/api/generate", json=payload, stream=True, timeout=timeout) as response:
                if response.status_code != 200:
                    error_text = f"Error: HTTP {response.status_code} - {response.text}"
                    if update_callback:
                        update_callback(model, error_text, 0, time.time() - start_time)
                    return {
                        "model": model,
                        "response": error_text,
                        "elapsed_time": 0,
                        "token_count": 0
                    }
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        # Parse the streaming response
                        json_line = line.decode('utf-8')
                        if not json_line.startswith('{'):
                            continue
                            
                        import json
                        chunk = json.loads(json_line)
                        
                        if 'response' in chunk:
                            token = chunk['response']
                            response_text += token
                            token_count += 1
                            
                            # Call the update callback with latest text
                            if update_callback and token_count % 5 == 0:  # Update every 5 tokens
                                current_time = time.time() - start_time
                                update_callback(model, response_text, token_count, current_time)
                        
                        # Check if we're done
                        if chunk.get('done', False):
                            break
                            
                    except Exception as e:
                        print(f"Error parsing chunk: {e}")
                        continue
                
                elapsed_time = time.time() - start_time
                
                # Final update
                if update_callback:
                    update_callback(model, response_text, token_count, elapsed_time)
                
                return {
                    "model": model,
                    "response": response_text,
                    "elapsed_time": round(elapsed_time, 2),
                    "token_count": token_count
                }
                
        except requests.exceptions.Timeout:
            error_text = f"Error: Request timed out after {timeout} seconds. The model might be too large or unavailable."
            if update_callback:
                update_callback(model, error_text, 0, time.time() - start_time)
            return {
                "model": model,
                "response": error_text,
                "elapsed_time": time.time() - start_time,
                "token_count": token_count
            }
        except Exception as e:
            error_text = f"Error: {str(e)}"
            if update_callback:
                update_callback(model, error_text, 0, time.time() - start_time)
            return {
                "model": model,
                "response": error_text,
                "elapsed_time": time.time() - start_time,
                "token_count": token_count
            }


def get_theme():
    return gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
        text_size=gr.themes.sizes.text_md,
    )


def create_app():
    """Create and configure the Gradio app"""
    ollama_api = OllamaAPI()
    available_models = ollama_api.get_available_models()
    
    # Create shared state variables for thread communication
    model1_response = {"text": "", "tokens": 0, "time": 0.0}
    model2_response = {"text": "", "tokens": 0, "time": 0.0}
    all_results = {}
    
    # Create locks for thread safety
    model1_lock = threading.Lock()
    model2_lock = threading.Lock()
    results_lock = threading.Lock()
    
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
                    value=available_models[1] if len(available_models) >= 2 else None
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
                    download_btn = gr.Button("Download Results")
        
        # Side-by-side comparison area
        with gr.Row(equal_height=True):
            # Left model output
            with gr.Column():
                model1_info = gr.Markdown("### Model 1")
                model1_stats = gr.Markdown("Time: 0.00s | Tokens: 0")
                model1_output = gr.Textbox(
                    label="Output",
                    elem_id="model1-output",
                    elem_classes=["model-output"],
                    lines=20,
                    max_lines=30,
                    show_copy_button=True,
                    container=False
                )
            
            # Right model output
            with gr.Column():
                model2_info = gr.Markdown("### Model 2")
                model2_stats = gr.Markdown("Time: 0.00s | Tokens: 0")
                model2_output = gr.Textbox(
                    label="Output",
                    elem_id="model2-output",
                    elem_classes=["model-output"],
                    lines=20,
                    max_lines=30,
                    show_copy_button=True,
                    container=False
                )
        
        status = gr.Markdown("Ready to compare models")
        results_file = gr.File(label="Download Results", visible=False)
        
        # Add custom CSS for better formatting
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
        
        def update_model_list():
            models = ollama_api.get_available_models()
            if not models:
                return gr.Dropdown.update(choices=[], value=None), gr.Dropdown.update(choices=[], value=None), "No models found. Is Ollama running? Try starting it with 'ollama serve'"
            return gr.Dropdown.update(choices=models, value=models[0] if models else None), gr.Dropdown.update(choices=models, value=models[1] if len(models) >= 2 else None), f"Found {len(models)} models"
        
        def update_custom_models(dropdown1_value, dropdown2_value, custom_models_text):
            if not custom_models_text:
                return dropdown1_value, dropdown2_value, ""
            
            custom_model_list = [m.strip() for m in custom_models_text.split(",") if m.strip()]
            
            # No need to update if no custom models specified
            if not custom_model_list:
                return dropdown1_value, dropdown2_value, ""
            
            # Get existing models
            existing_models = ollama_api.get_available_models()
            
            # Add custom models to the list
            all_models = existing_models.copy()
            for m in custom_model_list:
                if m not in all_models:
                    all_models.append(m)
            
            # Update dropdowns with combined list
            return gr.Dropdown.update(choices=all_models, value=dropdown1_value), gr.Dropdown.update(choices=all_models, value=dropdown2_value), f"Added custom models: {', '.join(custom_model_list)}"
        
        def clear_outputs():
            # Reset global state
            nonlocal model1_response, model2_response, all_results
            with model1_lock:
                model1_response = {"text": "", "tokens": 0, "time": 0.0}
            with model2_lock:
                model2_response = {"text": "", "tokens": 0, "time": 0.0}
            with results_lock:
                all_results = {}
            
            return "### Model 1", "### Model 2", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Ready to compare models", None
        
        def download_results():
            with results_lock:
                if not all_results:
                    return None
                
                try:
                    # Create a JSON file with the results
                    results_json = json.dumps(all_results, indent=2)
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"ollama_comparison_{timestamp}.json"
                    
                    # Return the file data
                    return gr.File.update(value=results_json, visible=True, label=f"Download Results ({filename})")
                except Exception as e:
                    print(f"Error creating results file: {e}")
                    return None
        
        def update_ui():
            """Function to update UI based on current state from model threads"""
            # Access the global state
            nonlocal model1_response, model2_response, all_results
            
            # Copy the current state safely
            with model1_lock:
                m1_resp = model1_response.copy()
            with model2_lock:
                m2_resp = model2_response.copy()
            with results_lock:
                results_copy = all_results.copy()
            
            # Update the UI
            return (
                f"### {m1_resp.get('model', 'Model 1')}",
                f"### {m2_resp.get('model', 'Model 2')}",
                f"Time: {m1_resp['time']:.2f}s | Tokens: {m1_resp['tokens']}",
                f"Time: {m2_resp['time']:.2f}s | Tokens: {m2_resp['tokens']}",
                m1_resp['text'],
                m2_resp['text'],
                "Models running in parallel... Responses will appear as they're generated."
            )
        
        def compare_models(model1_name, model2_name, sys_prompt, user_prompt, temp, max_tok, timeout_sec):
            # Access shared state
            nonlocal model1_response, model2_response, all_results
            
            # Clear previous outputs
            with model1_lock:
                model1_response = {"text": "", "tokens": 0, "time": 0.0, "model": model1_name}
            with model2_lock:
                model2_response = {"text": "", "tokens": 0, "time": 0.0, "model": model2_name}
            with results_lock:
                all_results = {}
            
            if not model1_name or not model2_name:
                return f"### {model1_name or 'Model 1'}", f"### {model2_name or 'Model 2'}", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Please select both models", None
            
            if not user_prompt:
                return f"### {model1_name}", f"### {model2_name}", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Please enter a prompt", None
            
            # Create callbacks for updating model outputs
            def update_model1_callback(model_name, text, tokens, time_elapsed):
                # Update the global state with thread safety
                nonlocal model1_response, all_results
                with model1_lock:
                    model1_response = {
                        "text": text,
                        "tokens": tokens,
                        "time": time_elapsed,
                        "model": model_name
                    }
                
                with results_lock:
                    all_results[model_name] = {
                        "response": text,
                        "elapsed_time": round(time_elapsed, 2),
                        "token_count": tokens
                    }
            
            def update_model2_callback(model_name, text, tokens, time_elapsed):
                # Update the global state with thread safety
                nonlocal model2_response, all_results
                with model2_lock:
                    model2_response = {
                        "text": text,
                        "tokens": tokens,
                        "time": time_elapsed,
                        "model": model_name
                    }
                
                with results_lock:
                    all_results[model_name] = {
                        "response": text,
                        "elapsed_time": round(time_elapsed, 2),
                        "token_count": tokens
                    }
            
            # Start threads for each model
            def run_model1():
                ollama_api.generate_streaming(
                    model1_name, user_prompt, sys_prompt, temp, max_tok,
                    update_callback=update_model1_callback,
                    timeout=timeout_sec
                )
            
            def run_model2():
                ollama_api.generate_streaming(
                    model2_name, user_prompt, sys_prompt, temp, max_tok,
                    update_callback=update_model2_callback,
                    timeout=timeout_sec
                )
            
            # Start threads
            thread1 = threading.Thread(target=run_model1)
            thread2 = threading.Thread(target=run_model2)
            
            thread1.start()
            thread2.start()
            
            # Set up UI update timer
            timer = gr.Timer(0.5, update_ui, [], [model1_info, model2_info, model1_stats, model2_stats, model1_output, model2_output, status])
            timer.start()
            
            return f"### {model1_name}", f"### {model2_name}", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Models running in parallel... Responses will appear as they're generated.", None
        
        # Set up event handlers
        refresh_btn.click(
            update_model_list,
            outputs=[model1, model2, status]
        )
        
        custom_models.change(
            update_custom_models,
            inputs=[model1, model2, custom_models],
            outputs=[model1, model2, status]
        )
        
        generate_btn.click(
            compare_models,
            inputs=[
                model1, 
                model2, 
                system_prompt, 
                prompt, 
                temperature, 
                max_tokens,
                timeout
            ],
            outputs=[
                model1_info, 
                model2_info, 
                model1_stats, 
                model2_stats,
                model1_output, 
                model2_output, 
                status,
                results_file
            ]
        )
        
        clear_btn.click(
            clear_outputs,
            outputs=[
                model1_info, 
                model2_info, 
                model1_stats, 
                model2_stats,
                model1_output, 
                model2_output, 
                status,
                results_file
            ]
        )
        
        download_btn.click(
            download_results,
            outputs=[results_file]
        )
        
        # Custom CSS
        app.load(
            None,
            js="""
            () => {
                // Add custom CSS for better side-by-side comparison
                const style = document.createElement('style');
                style.textContent = `
                    .model-output textarea {
                        min-height: 500px;
                        border: 1px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 16px;
                        overflow-y: auto;
                        background-color: #f9f9f9;
                        font-family: monospace;
                        font-size: 14px;
                        line-height: 1.5;
                    }
                `;
                document.head.appendChild(style);
            }
            """
        )
        
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()