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
                    container=False,
                    interactive=False
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
                    container=False,
                    interactive=False
                )
        
        status = gr.Markdown("Ready to compare models")
        results_file = gr.File(label="Download Results", visible=False)
        running_threads = gr.State([])
        model_results = gr.State({})
        
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
                return {
                    model1: gr.Dropdown(choices=[], value=None),
                    model2: gr.Dropdown(choices=[], value=None),
                    status: "No models found. Is Ollama running? Try starting it with 'ollama serve'"
                }
            return {
                model1: gr.Dropdown(choices=models, value=models[0] if models else None),
                model2: gr.Dropdown(choices=models, value=models[1] if len(models) >= 2 else None),
                status: f"Found {len(models)} models"
            }
        
        def update_custom_models(dropdown1_value, dropdown2_value, custom_models_text):
            if not custom_models_text:
                return {model1: dropdown1_value, model2: dropdown2_value}
            
            custom_model_list = [m.strip() for m in custom_models_text.split(",") if m.strip()]
            
            # No need to update if no custom models specified
            if not custom_model_list:
                return {model1: dropdown1_value, model2: dropdown2_value}
            
            # Get existing models
            existing_models = ollama_api.get_available_models()
            
            # Add custom models to the list
            all_models = existing_models.copy()
            for m in custom_model_list:
                if m not in all_models:
                    all_models.append(m)
            
            # Update dropdowns with combined list
            return {
                model1: gr.Dropdown(choices=all_models, value=dropdown1_value),
                model2: gr.Dropdown(choices=all_models, value=dropdown2_value),
                status: f"Added custom models: {', '.join(custom_model_list)}"
            }
        
        def update_model_output(model_index, model_name, text, tokens, elapsed_time):
            if model_index == 0:
                return {
                    model1_info: f"### {model_name}",
                    model1_stats: f"Time: {elapsed_time:.2f}s | Tokens: {tokens}",
                    model1_output: text
                }
            else:
                return {
                    model2_info: f"### {model_name}",
                    model2_stats: f"Time: {elapsed_time:.2f}s | Tokens: {tokens}",
                    model2_output: text
                }
        
        def clear_outputs():
            return {
                model1_info: "### Model 1",
                model2_info: "### Model 2",
                model1_stats: "Time: 0.00s | Tokens: 0",
                model2_stats: "Time: 0.00s | Tokens: 0",
                model1_output: "",
                model2_output: "",
                status: "Ready to compare models",
                model_results: {},
                results_file: None
            }
        
        def download_results(results):
            if not results:
                return None
            
            try:
                # Create a JSON file with the results
                results_json = json.dumps(results, indent=2)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"ollama_comparison_{timestamp}.json"
                
                # Return the file data
                return gr.File.update(value=results_json, visible=True, label=f"Download Results ({filename})")
            except Exception as e:
                print(f"Error creating results file: {e}")
                return None
        
        def compare_models(model1_name, model2_name, sys_prompt, user_prompt, temp, tokens, timeout_sec, threads):
            # Stop any running threads
            for thread in threads:
                if thread.is_alive():
                    pass  # We can't really stop threads in Python, but this is a placeholder
            
            # Clear previous outputs
            yield {
                model1_info: f"### {model1_name}",
                model2_info: f"### {model2_name}",
                model1_stats: "Time: 0.00s | Tokens: 0",
                model2_stats: "Time: 0.00s | Tokens: 0",
                model1_output: "",
                model2_output: "",
                status: "Starting comparison...",
                running_threads: [],
                model_results: {},
                results_file: None
            }
            
            if not model1_name or not model2_name:
                yield {
                    status: "Please select both models",
                    running_threads: []
                }
                return
            
            if not user_prompt:
                yield {
                    status: "Please enter a prompt",
                    running_threads: []
                }
                return
            
            new_threads = []
            
            # Create callbacks for updating the UI
            def update_model1(model_name, text, tokens, time_elapsed):
                app.queue(lambda: update_model_output(0, model_name, text, tokens, time_elapsed))
            
            def update_model2(model_name, text, tokens, time_elapsed):
                app.queue(lambda: update_model_output(1, model_name, text, tokens, time_elapsed))
            
            # Create shared results dictionary
            all_results = {}
            
            # Start threads for each model
            def run_model1():
                result = ollama_api.generate_streaming(
                    model1_name, user_prompt, sys_prompt, temp, tokens,
                    update_callback=update_model1,
                    timeout=timeout_sec
                )
                # Store final result
                all_results[model1_name] = result
                app.queue(lambda: gr.State.update(value=all_results), outputs=[model_results])
            
            def run_model2():
                result = ollama_api.generate_streaming(
                    model2_name, user_prompt, sys_prompt, temp, tokens,
                    update_callback=update_model2,
                    timeout=timeout_sec
                )
                # Store final result
                all_results[model2_name] = result
                app.queue(lambda: gr.State.update(value=all_results), outputs=[model_results])
            
            # Start threads
            thread1 = threading.Thread(target=run_model1)
            thread2 = threading.Thread(target=run_model2)
            
            thread1.start()
            thread2.start()
            
            new_threads = [thread1, thread2]
            
            yield {
                status: "Models running in parallel... Responses will appear as they're generated.",
                running_threads: new_threads
            }
        
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
                timeout,
                running_threads
            ],
            outputs=[
                model1_info, 
                model2_info, 
                model1_stats, 
                model2_stats,
                model1_output, 
                model2_output, 
                status, 
                running_threads,
                model_results,
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
                model_results,
                results_file
            ]
        )
        
        download_btn.click(
            download_results,
            inputs=[model_results],
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