#!/usr/bin/env python3
"""
Ollama Model Comparator - Compare responses from different Ollama models
Simplified frontend with side-by-side outputs and continuous updates
"""

import gradio as gr
import requests
import time
import threading
from typing import List, Dict, Any

class OllamaAPI:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        
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
            # Use a longer timeout for models that take time to load
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


def main():
    # Create the Gradio interface
    with gr.Blocks(title="Ollama Model Comparator", theme=gr.themes.Base()) as app:
        gr.Markdown("# 🤖 Ollama Model Comparator")
        
        with gr.Row():
            # Left column for inputs
            with gr.Column():
                ollama_url = gr.Textbox(
                    label="Ollama API URL", 
                    value="http://localhost:11434",
                    placeholder="http://localhost:11434"
                )
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Models")
                    timeout_slider = gr.Slider(
                        minimum=30, 
                        maximum=600, 
                        step=30, 
                        value=180,
                        label="Request Timeout (seconds)"
                    )
                
                model_input1 = gr.Textbox(
                    label="Model 1",
                    placeholder="e.g., llama3:8b"
                )
                
                model_input2 = gr.Textbox(
                    label="Model 2",
                    placeholder="e.g., gemma:2b"
                )
                
                # Prompts
                system_prompt = gr.Textbox(
                    label="System Prompt (optional)", 
                    placeholder="You are a helpful assistant...",
                    lines=2
                )
                
                prompt = gr.Textbox(
                    label="Prompt", 
                    placeholder="Enter your prompt here...", 
                    lines=4
                )
                
                # Parameters
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.0, 
                        maximum=1.0, 
                        step=0.1, 
                        value=0.7, 
                        label="Temperature"
                    )
                    max_tokens = gr.Slider(
                        minimum=100, 
                        maximum=4000, 
                        step=100, 
                        value=1000, 
                        label="Max Tokens"
                    )
                
                # Action buttons
                with gr.Row():
                    generate_btn = gr.Button("Compare Models", variant="primary")
                    clear_btn = gr.Button("Clear Results")
                
                # Available models accordion
                with gr.Accordion("Available Models", open=False):
                    available_models = gr.Dataframe(
                        headers=["Model"],
                        datatype=["str"],
                        col_count=(1, "fixed"),
                        label="Click 'Refresh Models' to update"
                    )

        # Main comparison output area
        with gr.Row():
            # Model 1 output
            with gr.Column():
                model1_name = gr.Markdown("### Model 1")
                model1_stats = gr.Markdown("Time: 0.00s | Tokens: 0 | Speed: 0 t/s")
                model1_output = gr.Textbox(
                    label="Output",
                    lines=20,
                    max_lines=30,
                    autoscroll=True
                )
            
            # Model 2 output
            with gr.Column():
                model2_name = gr.Markdown("### Model 2")
                model2_stats = gr.Markdown("Time: 0.00s | Tokens: 0 | Speed: 0 t/s")
                model2_output = gr.Textbox(
                    label="Output",
                    lines=20,
                    max_lines=30,
                    autoscroll=True
                )
        
        # Status message
        status = gr.Markdown("Ready to compare models")
        
        # State variables
        running_threads = gr.State([])

        # Function to refresh model list
        def refresh_models(url):
            try:
                api = OllamaAPI(url)
                models = api.get_available_models()
                
                if not models:
                    return [[]], "No models found or couldn't connect to Ollama"
                
                # Format models as rows for the dataframe
                model_rows = [[model] for model in models]
                
                return model_rows, f"Found {len(models)} models"
            except Exception as e:
                return [[]], f"Error refreshing models: {str(e)}"

        # Function to update a model's output in real-time
        def update_model_output(model_index, model_name, text, tokens, elapsed_time):
            if model_index == 0:
                # Calculate tokens per second
                tokens_per_sec = round(tokens / elapsed_time, 1) if elapsed_time > 0 else 0
                return {
                    model1_name: f"### {model_name}",
                    model1_stats: f"Time: {elapsed_time:.2f}s | Tokens: {tokens} | Speed: {tokens_per_sec} t/s",
                    model1_output: text
                }
            else:
                # Calculate tokens per second
                tokens_per_sec = round(tokens / elapsed_time, 1) if elapsed_time > 0 else 0
                return {
                    model2_name: f"### {model_name}",
                    model2_stats: f"Time: {elapsed_time:.2f}s | Tokens: {tokens} | Speed: {tokens_per_sec} t/s",
                    model2_output: text
                }

        # Function to compare models
        def compare_models(api_url, model1, model2, system_prompt, user_prompt, temp, max_tokens, timeout_sec, threads):
            # Stop any running threads
            for thread in threads:
                if thread.is_alive():
                    # We can't forcefully stop threads in Python, but we can set a flag
                    # This is just a cleanup step
                    pass
            
            # Clear previous outputs
            yield {
                model1_name: f"### {model1}",
                model2_name: f"### {model2}",
                model1_stats: "Time: 0.00s | Tokens: 0 | Speed: 0 t/s",
                model2_stats: "Time: 0.00s | Tokens: 0 | Speed: 0 t/s",
                model1_output: "",
                model2_output: "",
                status: "Starting comparison...",
                running_threads: []
            }
            
            if not model1 or not model2:
                yield {
                    status: "Please enter both model names",
                    running_threads: []
                }
                return
            
            api = OllamaAPI(api_url)
            new_threads = []
            
            # Create thread for model 1
            def run_model1():
                api.generate_streaming(
                    model1, user_prompt, system_prompt, temp, max_tokens,
                    lambda text, tokens, time: app.queue(
                        lambda: update_model_output(0, model1, text, tokens, time)
                    ),
                    timeout=timeout_sec
                )
            
            # Create thread for model 2
            def run_model2():
                api.generate_streaming(
                    model2, user_prompt, system_prompt, temp, max_tokens,
                    lambda text, tokens, time: app.queue(
                        lambda: update_model_output(1, model2, text, tokens, time)
                    ),
                    timeout=timeout_sec
                )
            
            # Start threads
            thread1 = threading.Thread(target=run_model1)
            thread2 = threading.Thread(target=run_model2)
            
            thread1.start()
            thread2.start()
            
            new_threads = [thread1, thread2]
            
            yield {
                status: "Models running in parallel...",
                running_threads: new_threads
            }

        # Function to clear all outputs
        def clear_outputs():
            return {
                model1_name: "### Model 1",
                model2_name: "### Model 2",
                model1_stats: "Time: 0.00s | Tokens: 0 | Speed: 0 t/s",
                model2_stats: "Time: 0.00s | Tokens: 0 | Speed: 0 t/s",
                model1_output: "",
                model2_output: "",
                status: "Ready to compare models"
            }

        # Connect event handlers
        refresh_btn.click(
            refresh_models,
            inputs=[ollama_url],
            outputs=[available_models, status]
        )
        
        generate_btn.click(
            compare_models,
            inputs=[
                ollama_url,
                model_input1,
                model_input2,
                system_prompt,
                prompt,
                temperature,
                max_tokens,
                timeout_slider,
                running_threads
            ],
            outputs=[
                model1_name,
                model2_name,
                model1_stats,
                model2_stats,
                model1_output,
                model2_output,
                status,
                running_threads
            ]
        )
        
        clear_btn.click(
            clear_outputs,
            outputs=[
                model1_name,
                model2_name,
                model1_stats,
                model2_stats,
                model1_output,
                model2_output,
                status
            ]
        )
    
    # Launch the app
    app.launch()

if __name__ == "__main__":
    main()