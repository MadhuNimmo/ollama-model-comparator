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
                         update_fn=None, timeout=180):
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
                    if update_fn:
                        update_fn(model, error_text, 0, time.time() - start_time)
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
                            
                        chunk = json.loads(json_line)
                        
                        if 'response' in chunk:
                            token = chunk['response']
                            response_text += token
                            token_count += 1
                            
                            # Call the update function with latest text
                            if update_fn and token_count % 5 == 0:  # Update every 5 tokens
                                elapsed_time = time.time() - start_time
                                update_fn(model, response_text, token_count, elapsed_time)
                        
                        # Check if we're done
                        if chunk.get('done', False):
                            break
                            
                    except Exception as e:
                        print(f"Error parsing chunk: {e}")
                        continue
                
                elapsed_time = time.time() - start_time
                
                # Final update
                if update_fn:
                    update_fn(model, response_text, token_count, elapsed_time)
                
                return {
                    "model": model,
                    "response": response_text,
                    "elapsed_time": round(elapsed_time, 2),
                    "token_count": token_count
                }
                
        except requests.exceptions.Timeout:
            error_text = f"Error: Request timed out after {timeout} seconds. The model might be too large or unavailable."
            if update_fn:
                update_fn(model, error_text, 0, time.time() - start_time)
            return {
                "model": model,
                "response": error_text,
                "elapsed_time": time.time() - start_time,
                "token_count": token_count
            }
        except Exception as e:
            error_text = f"Error: {str(e)}"
            if update_fn:
                update_fn(model, error_text, 0, time.time() - start_time)
            return {
                "model": model,
                "response": error_text,
                "elapsed_time": time.time() - start_time,
                "token_count": token_count
            }


def create_app():
    """Create and configure the Gradio app"""
    ollama_api = OllamaAPI()
    available_models = ollama_api.get_available_models()
    
    with gr.Blocks(title="Ollama Model Comparator", theme=gr.themes.Soft(primary_hue="blue")) as app:
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
        
        # Side-by-side comparison area
        with gr.Row(equal_height=True):
            # Left model output
            with gr.Column():
                model1_info = gr.Markdown("### Model 1")
                model1_stats = gr.Markdown("Time: 0.00s | Tokens: 0")
                model1_output = gr.Markdown(
                    label="",
                    elem_id="model1-output",
                    elem_classes=["model-output"],
                )
            
            # Right model output
            with gr.Column():
                model2_info = gr.Markdown("### Model 2")
                model2_stats = gr.Markdown("Time: 0.00s | Tokens: 0")
                model2_output = gr.Markdown(
                    label="",
                    elem_id="model2-output",
                    elem_classes=["model-output"],
                )
        
        status = gr.Markdown("Ready to compare models")
        
        # Shared state for output updates
        shared_state = gr.State({
            "model1_name": "",
            "model2_name": "",
            "model1_text": "",
            "model2_text": "",
            "model1_tokens": 0,
            "model2_tokens": 0,
            "model1_time": 0,
            "model2_time": 0
        })
        
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
            
            gr.Markdown("### Share Link")
            share_link = gr.Textbox(label="Share this app", value="To create a shareable link, restart with share=True")
        
        def update_model_list():
            models = ollama_api.get_available_models()
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
        
        def update_ui_with_state(state):
            """Update the UI with the current state values"""
            return {
                model1_info: f"### {state['model1_name']}",
                model1_stats: f"Time: {state['model1_time']:.2f}s | Tokens: {state['model1_tokens']}",
                model1_output: state['model1_text'],
                model2_info: f"### {state['model2_name']}",
                model2_stats: f"Time: {state['model2_time']:.2f}s | Tokens: {state['model2_tokens']}",
                model2_output: state['model2_text']
            }
        
        def clear_outputs():
            """Reset all outputs"""
            new_state = {
                "model1_name": "Model 1",
                "model2_name": "Model 2",
                "model1_text": "",
                "model2_text": "",
                "model1_tokens": 0,
                "model2_tokens": 0,
                "model1_time": 0.0,
                "model2_time": 0.0
            }
            
            result = update_ui_with_state(new_state)
            result[status] = "Ready to compare models"
            result[shared_state] = new_state
            
            return result
        
        def compare_models(model1_name, model2_name, sys_prompt, user_prompt, temp, tokens, timeout_sec, curr_state):
            """Compare two models with periodic updates"""
            if not model1_name or not model2_name:
                return {
                    status: "Please select both models",
                    shared_state: curr_state
                }
            
            # Initialize the state with model names
            new_state = curr_state.copy()
            new_state.update({
                "model1_name": model1_name,
                "model2_name": model2_name,
                "model1_text": "",
                "model2_text": "",
                "model1_tokens": 0,
                "model2_tokens": 0,
                "model1_time": 0.0,
                "model2_time": 0.0
            })
            
            # First update to clear previous results
            result = update_ui_with_state(new_state)
            result[status] = "Starting comparison..."
            result[shared_state] = new_state
            
            yield result
            
            # Create event to track when both models are done
            model1_done = threading.Event()
            model2_done = threading.Event()
            lock = threading.Lock()
            
            # Define update functions for each model
            def update_model1(model_name, text, tokens, elapsed_time):
                nonlocal new_state
                with lock:
                    # Update the state with new data
                    state_copy = new_state.copy()
                    state_copy["model1_text"] = text
                    state_copy["model1_tokens"] = tokens
                    state_copy["model1_time"] = elapsed_time
                    new_state = state_copy
                    
                    # Create the UI update
                    result = {
                        model1_info: f"### {model_name}",
                        model1_stats: f"Time: {elapsed_time:.2f}s | Tokens: {tokens}",
                        model1_output: text,
                        shared_state: new_state
                    }
                    
                    # If both models are done, update the status
                    if model1_done.is_set() and model2_done.is_set():
                        result[status] = "Completed generating responses."
                    
                    yield result
            
            def update_model2(model_name, text, tokens, elapsed_time):
                nonlocal new_state
                with lock:
                    # Update the state with new data
                    state_copy = new_state.copy()
                    state_copy["model2_text"] = text
                    state_copy["model2_tokens"] = tokens
                    state_copy["model2_time"] = elapsed_time
                    new_state = state_copy
                    
                    # Create the UI update
                    result = {
                        model2_info: f"### {model_name}",
                        model2_stats: f"Time: {elapsed_time:.2f}s | Tokens: {tokens}",
                        model2_output: text,
                        shared_state: new_state
                    }
                    
                    # If both models are done, update the status
                    if model1_done.is_set() and model2_done.is_set():
                        result[status] = "Completed generating responses."
                    
                    yield result
            
            # Run models in separate threads with synchronous updates
            def run_model1():
                try:
                    def model1_update_fn(model_name, text, tokens, elapsed_time):
                        gr.Markdown.update(f"### {model_name}")
                        gr.Markdown.update(f"Time: {elapsed_time:.2f}s | Tokens: {tokens}")
                        gr.Markdown.update(text)
                        
                    ollama_api.generate_streaming(
                        model1_name, user_prompt, sys_prompt, temp, tokens,
                        lambda model, text, tokens, time: app.update(
                            fn=lambda: {
                                model1_info: f"### {model}",
                                model1_stats: f"Time: {time:.2f}s | Tokens: {tokens}",
                                model1_output: text
                            }
                        ),
                        timeout=timeout_sec
                    )
                finally:
                    model1_done.set()
            
            def run_model2():
                try:
                    ollama_api.generate_streaming(
                        model2_name, user_prompt, sys_prompt, temp, tokens,
                        lambda model, text, tokens, time: app.update(
                            fn=lambda: {
                                model2_info: f"### {model}",
                                model2_stats: f"Time: {time:.2f}s | Tokens: {tokens}",
                                model2_output: text
                            }
                        ),
                        timeout=timeout_sec
                    )
                finally:
                    model2_done.set()
            
            # Start threads
            thread1 = threading.Thread(target=run_model1)
            thread2 = threading.Thread(target=run_model2)
            
            thread1.daemon = True
            thread2.daemon = True
            
            thread1.start()
            thread2.start()
            
            yield {
                status: "Models running in parallel... Responses will appear as they're generated.",
                shared_state: new_state
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
                shared_state
            ],
            outputs=[
                model1_info, 
                model2_info, 
                model1_stats, 
                model2_stats,
                model1_output, 
                model2_output, 
                status,
                shared_state
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
                shared_state
            ]
        )
        
        # Custom CSS
        app.load(
            lambda: {},
            js="""
            () => {
                // Add custom CSS for better side-by-side comparison
                const style = document.createElement('style');
                style.textContent = `
                    .model-output {
                        min-height: 500px;
                        border: 1px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 16px;
                        overflow-y: auto;
                        background-color: #f9f9f9;
                    }
                `;
                document.head.appendChild(style);
            }
            """
        )
        
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(debug=True)  # Set share=True to create a public link