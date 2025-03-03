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
    
    # Shared state for model responses
    model1_text = ""
    model2_text = ""
    model1_tokens = 0
    model2_tokens = 0
    model1_time = 0.0
    model2_time = 0.0
    all_results = {}
    
    # Create locks for thread safety
    model1_lock = threading.Lock()
    model2_lock = threading.Lock()
    results_lock = threading.Lock()
    
    with gr.Blocks(title="Ollama Model Comparator", theme=get_theme()) as app:
        gr.Markdown("# 🤖 Ollama Model Comparator")
        
        # Create a state to track if models are currently generating
        generating = gr.State(False)
        
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
            """Refresh the model list from Ollama"""
            models = ollama_api.get_available_models()
            if not models:
                return [], [], "No models found. Is Ollama running? Try starting it with 'ollama serve'"
            
            model1_val = models[0] if models else None
            model2_val = models[1] if len(models) > 1 else None
            
            return models, models, f"Found {len(models)} models", model1_val, model2_val
        
        def update_custom_models(custom_text, current_model1, current_model2):
            """Add custom models to the dropdown list"""
            if not custom_text:
                return [], "", current_model1, current_model2
            
            custom_models = [m.strip() for m in custom_text.split(",") if m.strip()]
            if not custom_models:
                return [], "", current_model1, current_model2
            
            # Combine with existing models
            existing_models = ollama_api.get_available_models()
            all_models = existing_models.copy()
            
            for m in custom_models:
                if m not in all_models:
                    all_models.append(m)
            
            return all_models, f"Added custom models: {', '.join(custom_models)}", current_model1, current_model2
        
        def clear_outputs():
            """Clear all outputs and reset state"""
            nonlocal model1_text, model2_text, model1_tokens, model2_tokens, model1_time, model2_time, all_results
            
            with model1_lock:
                model1_text = ""
                model1_tokens = 0
                model1_time = 0.0
            
            with model2_lock:
                model2_text = ""
                model2_tokens = 0
                model2_time = 0.0
            
            with results_lock:
                all_results = {}
            
            return "### Model 1", "### Model 2", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Ready to compare models", None, False
        
        def download_results():
            """Prepare results for download"""
            with results_lock:
                if not all_results:
                    return None
                
                try:
                    # Create a JSON file with the results
                    results_json = json.dumps(all_results, indent=2)
                    return results_json
                except Exception as e:
                    print(f"Error creating results file: {e}")
                    return None
        
        def check_for_updates():
            """Check for updates from model threads and update the UI"""
            # Read the current state (thread-safe)
            with model1_lock:
                m1_text = model1_text
                m1_tokens = model1_tokens
                m1_time = model1_time
            
            with model2_lock:
                m2_text = model2_text
                m2_tokens = model2_tokens
                m2_time = model2_time
            
            # Update UI components
            return (
                m1_text,
                m2_text,
                f"Time: {m1_time:.2f}s | Tokens: {m1_tokens}",
                f"Time: {m2_time:.2f}s | Tokens: {m2_tokens}"
            )
        
        def compare_models(model1_name, model2_name, sys_prompt, user_prompt, temp, max_tok, timeout_sec):
            """Start model comparison"""
            nonlocal model1_text, model2_text, model1_tokens, model2_tokens, model1_time, model2_time, all_results
            
            # Reset state
            with model1_lock:
                model1_text = ""
                model1_tokens = 0
                model1_time = 0.0
            
            with model2_lock:
                model2_text = ""
                model2_tokens = 0
                model2_time = 0.0
            
            with results_lock:
                all_results = {}
            
            # Validate inputs
            if not model1_name or not model2_name:
                return f"### {model1_name or 'Model 1'}", f"### {model2_name or 'Model 2'}", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Please select both models", None, False
            
            if not user_prompt:
                return f"### {model1_name}", f"### {model2_name}", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Please enter a prompt", None, False
            
            # Define callback functions for model responses
            def update_model1(model_name, text, tokens, time_elapsed):
                nonlocal model1_text, model1_tokens, model1_time, all_results
                
                with model1_lock:
                    model1_text = text
                    model1_tokens = tokens
                    model1_time = time_elapsed
                
                with results_lock:
                    all_results[model_name] = {
                        "response": text,
                        "elapsed_time": round(time_elapsed, 2),
                        "token_count": tokens
                    }
            
            def update_model2(model_name, text, tokens, time_elapsed):
                nonlocal model2_text, model2_tokens, model2_time, all_results
                
                with model2_lock:
                    model2_text = text
                    model2_tokens = tokens
                    model2_time = time_elapsed
                
                with results_lock:
                    all_results[model_name] = {
                        "response": text,
                        "elapsed_time": round(time_elapsed, 2),
                        "token_count": tokens
                    }
            
            # Start threads for each model
            def run_model1():
                try:
                    ollama_api.generate_streaming(
                        model1_name, user_prompt, sys_prompt, temp, max_tok,
                        update_callback=update_model1,
                        timeout=timeout_sec
                    )
                except Exception as e:
                    print(f"Error in model1 thread: {e}")
                    update_model1(model1_name, f"Error: {str(e)}", 0, 0)
            
            def run_model2():
                try:
                    ollama_api.generate_streaming(
                        model2_name, user_prompt, sys_prompt, temp, max_tok,
                        update_callback=update_model2,
                        timeout=timeout_sec
                    )
                except Exception as e:
                    print(f"Error in model2 thread: {e}")
                    update_model2(model2_name, f"Error: {str(e)}", 0, 0)
            
            # Start threads
            threading.Thread(target=run_model1, daemon=True).start()
            threading.Thread(target=run_model2, daemon=True).start()
            
            # Set initial UI state
            return f"### {model1_name}", f"### {model2_name}", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Models running in parallel... Responses will appear as they're generated.", None, True
        
        # Set up event handlers
        refresh_btn.click(
            fn=update_model_list,
            outputs=[model1, model2, status, model1, model2]
        )
        
        custom_models.change(
            fn=update_custom_models,
            inputs=[custom_models, model1, model2],
            outputs=[model1, model2, status, model1, model2]
        )
        
        generate_btn.click(
            fn=compare_models,
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
                results_file,
                generating
            ]
        )
        
        clear_btn.click(
            fn=clear_outputs,
            outputs=[
                model1_info, 
                model2_info, 
                model1_stats, 
                model2_stats,
                model1_output, 
                model2_output, 
                status,
                results_file,
                generating
            ]
        )
        
        download_btn.click(
            fn=download_results,
            outputs=[results_file]
        )
        
        # Setup polling for updates when generating is True
        generating.change(
            fn=lambda x: 0.5 if x else None,  # Return the poll interval when generating is True
            inputs=[generating],
            outputs=[gr.Number(value=0, visible=False)]  # Hidden element to control polling
        )
        
        # Set up polling for model updates
        app.load(
            fn=lambda: 0,  # Dummy function
            inputs=None,
            outputs=[gr.Number(value=0, visible=False)],
            every=0.5,
            show_progress=False,
        ).then(
            fn=check_for_updates,
            inputs=None,
            outputs=[model1_output, model2_output, model1_stats, model2_stats],
            _js="""
            function(v) {
                // Only update if the model is generating (checked in JavaScript)
                const generatingEl = document.querySelector('[data-testid="generating"]');
                if (generatingEl && generatingEl.value === 'true') {
                    return [];  // Return empty to trigger the update
                }
                return null;  // Return null to cancel
            }
            """
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