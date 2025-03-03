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
    
    # Global variables for model responses
    responses = {
        "model1": {"text": "", "tokens": 0, "time": 0.0, "name": "Model 1"},
        "model2": {"text": "", "tokens": 0, "time": 0.0, "name": "Model 2"}
    }
    
    # Locks for thread safety
    response_lock = threading.Lock()
    
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
        
        # This doesn't rely on polling at all - just direct UI updates from the thread
        def update_ui():
            with response_lock:
                # Simple JS to inject into the page that updates the UI every second
                js_code = """
                <script>
                function setupUIUpdater() {
                    // Function to update UI elements
                    function updateUI() {
                        // Model 1 output and stats
                        const model1Output = document.querySelector('#model1-output textarea');
                        const model1Stats = document.querySelector('#model1-info + div .prose');
                        const model1Title = document.querySelector('#model1-info .prose');
                        
                        // Model 2 output and stats
                        const model2Output = document.querySelector('#model2-output textarea');
                        const model2Stats = document.querySelector('#model2-info + div .prose');
                        const model2Title = document.querySelector('#model2-info .prose');
                        
                        // Make an API call to get the latest data
                        fetch('/latest_updates')
                            .then(response => response.json())
                            .then(data => {
                                // Update model1 info
                                if (model1Output && data.model1) {
                                    model1Output.value = data.model1.text;
                                    model1Stats.innerHTML = `Time: ${data.model1.time.toFixed(2)}s | Tokens: ${data.model1.tokens}`;
                                    model1Title.innerHTML = `### ${data.model1.name}`;
                                }
                                
                                // Update model2 info
                                if (model2Output && data.model2) {
                                    model2Output.value = data.model2.text;
                                    model2Stats.innerHTML = `Time: ${data.model2.time.toFixed(2)}s | Tokens: ${data.model2.tokens}`;
                                    model2Title.innerHTML = `### ${data.model2.name}`;
                                }
                            })
                            .catch(error => console.error('Error fetching updates:', error));
                    }
                    
                    // Update the UI every second
                    setInterval(updateUI, 1000);
                }
                
                // Run the setup function when the page loads
                window.addEventListener('load', setupUIUpdater);
                </script>
                """
                
                return js_code
        
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
        
        def refresh_models():
            """Refresh the model list from Ollama"""
            models = ollama_api.get_available_models()
            if not models:
                return models, models, "No models found. Is Ollama running?"
            
            return models, models, f"Found {len(models)} models"
        
        def add_custom_models(custom_text):
            """Add custom models to the dropdown list"""
            if not custom_text:
                return [], ""
            
            custom_models = [m.strip() for m in custom_text.split(",") if m.strip()]
            if not custom_models:
                return [], ""
            
            # Combine with existing models
            existing_models = ollama_api.get_available_models()
            all_models = existing_models.copy()
            
            for m in custom_models:
                if m not in all_models:
                    all_models.append(m)
            
            return all_models, f"Added custom models: {', '.join(custom_models)}"
        
        def clear_outputs():
            """Clear all outputs and reset state"""
            with response_lock:
                responses["model1"] = {"text": "", "tokens": 0, "time": 0.0, "name": "Model 1"}
                responses["model2"] = {"text": "", "tokens": 0, "time": 0.0, "name": "Model 2"}
            
            return "### Model 1", "### Model 2", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Ready to compare models", None
        
        def download_results():
            """Prepare results for download"""
            with response_lock:
                results = {
                    responses["model1"]["name"]: {
                        "response": responses["model1"]["text"],
                        "elapsed_time": responses["model1"]["time"],
                        "token_count": responses["model1"]["tokens"]
                    },
                    responses["model2"]["name"]: {
                        "response": responses["model2"]["text"],
                        "elapsed_time": responses["model2"]["time"],
                        "token_count": responses["model2"]["tokens"]
                    }
                }
                
                try:
                    # Create a JSON file with the results
                    results_json = json.dumps(results, indent=2)
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"ollama_comparison_{timestamp}.json"
                    
                    return results_json
                except Exception as e:
                    print(f"Error creating results file: {e}")
                    return None
        
        def compare_models(model1_name, model2_name, sys_prompt, user_prompt, temp, max_tok, timeout_sec):
            """Start model comparison"""
            # Reset state
            with response_lock:
                responses["model1"] = {"text": "", "tokens": 0, "time": 0.0, "name": model1_name}
                responses["model2"] = {"text": "", "tokens": 0, "time": 0.0, "name": model2_name}
            
            # Validate inputs
            if not model1_name or not model2_name:
                return f"### {model1_name or 'Model 1'}", f"### {model2_name or 'Model 2'}", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Please select both models", None
            
            if not user_prompt:
                return f"### {model1_name}", f"### {model2_name}", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Please enter a prompt", None
                
            # Define callback functions for model responses
            def update_model1(model_name, text, tokens, time_elapsed):
                with response_lock:
                    responses["model1"]["text"] = text
                    responses["model1"]["tokens"] = tokens
                    responses["model1"]["time"] = time_elapsed
                    responses["model1"]["name"] = model_name
                    
                    # Gradio doesn't support updating UI from threads,
                    # but we're storing the results so they can be accessed via API
            
            def update_model2(model_name, text, tokens, time_elapsed):
                with response_lock:
                    responses["model2"]["text"] = text
                    responses["model2"]["tokens"] = tokens
                    responses["model2"]["time"] = time_elapsed
                    responses["model2"]["name"] = model_name
            
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
            
            # Set up manual refresh button to check progress
            return f"### {model1_name}", f"### {model2_name}", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Models running in parallel... Responses will appear below (manually refresh to update)", None
        
        # Simple way to get the latest state when user clicks refresh
        def get_latest_updates():
            with response_lock:
                m1 = responses["model1"]
                m2 = responses["model2"]
            
            return (
                f"### {m1['name']}",
                f"### {m2['name']}",
                f"Time: {m1['time']:.2f}s | Tokens: {m1['tokens']}",
                f"Time: {m2['time']:.2f}s | Tokens: {m2['tokens']}",
                m1['text'],
                m2['text']
            )
        
        # Manual refresh button
        refresh_outputs_btn = gr.Button("↻ Refresh Outputs")
        
        # Set up event handlers
        refresh_btn.click(
            fn=refresh_models,
            outputs=[model1.choices, model2.choices, status]
        )
        
        refresh_outputs_btn.click(
            fn=get_latest_updates,
            outputs=[model1_info, model2_info, model1_stats, model2_stats, model1_output, model2_output]
        )
        
        custom_models.change(
            fn=add_custom_models,
            inputs=[custom_models],
            outputs=[model1.choices, status]
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
                results_file
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
                results_file
            ]
        )
        
        download_btn.click(
            fn=download_results,
            outputs=[results_file]
        )
        
        # Custom CSS
        css = """
        <style>
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
        </style>
        """
        gr.HTML(css)
        
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()