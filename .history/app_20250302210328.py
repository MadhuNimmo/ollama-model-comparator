#!/usr/bin/env python3
"""
Ollama Model Comparator - Compare responses from different Ollama models
"""

import gradio as gr
import requests
import time
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
    
    def generate_streaming(
        self, 
        model, 
        prompt, 
        system_prompt="", 
        temperature=0.7, 
        max_tokens=1000, 
        timeout=180
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
            "stream": True  # Enable streaming
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
                        # Parse the streaming response
                        json_line = line.decode('utf-8', errors='ignore')
                        if not json_line.startswith('{'):
                            continue
                            
                        chunk = json.loads(json_line)
                        
                        if 'response' in chunk:
                            token = chunk['response']
                            response_text += token
                            token_count += 1
                            # Yield partial progress
                            yield response_text, token_count, time.time() - start_time
                        
                        if chunk.get('done', False):
                            # Streaming complete
                            return
                            
                    except Exception as e:
                        print(f"Error parsing chunk: {e}")
                        continue
                
        except requests.exceptions.Timeout:
            error_text = (
                f"Error: Request timed out after {timeout} seconds. "
                "The model might be too large or unavailable."
            )
            yield error_text, token_count, time.time() - start_time
        except Exception as e:
            error_text = f"Error: {str(e)}"
            yield error_text, token_count, time.time() - start_time


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
            
            1. Select two models to compare.
            2. Enter your prompt (and optional system prompt).
            3. Adjust temperature, max tokens, and timeout if needed.
            4. Click "Compare Models" to see results side by side.
            
            ### Notes
            - Models must be installed in Ollama before they can be compared.
            - If you don't see your models, click "Refresh Model List."
            - For large models, you may need to increase the timeout.
            - Responses are streamed in real-time as they're generated.
            """)
            
            gr.Markdown("### Share Link")
            share_link = gr.Textbox(
                label="Share this app", 
                value="To create a shareable link, restart with share=True"
            )
        
        def update_model_list():
            models = ollama_api.get_available_models()
            # Update dropdowns with newly fetched models
            return {
                model1: gr.Dropdown.update(choices=models, value=models[0] if models else None),
                model2: gr.Dropdown.update(
                    choices=models, 
                    value=models[1] if len(models) >= 2 else (models[0] if models else None)
                ),
                status: f"Found {len(models)} models"
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
        
        def compare_models(
            model1_name, 
            model2_name, 
            sys_prompt, 
            user_prompt, 
            temp, 
            tokens, 
            timeout_sec, 
            curr_state
        ):
            """Compare two models sequentially with streaming updates."""
            if not model1_name or not model2_name:
                yield {
                    status: "Please select both models",
                    shared_state: curr_state
                }
                return
            
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
            
            # 1) Update UI to show reset
            result = update_ui_with_state(new_state)
            result[status] = "Starting comparison (Model 1 first)..."
            result[shared_state] = new_state
            yield result
            
            # --- Generate with Model 1 ---
            text_so_far = ""
            tokens_so_far = 0
            
            for partial_text, partial_tokens, elapsed in ollama_api.generate_streaming(
                model1_name, 
                user_prompt, 
                system_prompt=sys_prompt, 
                temperature=temp, 
                max_tokens=tokens,
                timeout=timeout_sec
            ):
                text_so_far = partial_text
                tokens_so_far = partial_tokens
                new_state["model1_text"] = text_so_far
                new_state["model1_tokens"] = tokens_so_far
                new_state["model1_time"] = elapsed
                
                result = update_ui_with_state(new_state)
                result[shared_state] = new_state
                result[status] = f"Generating response from {model1_name}..."
                yield result
            
            # 2) Model 1 done, proceed with Model 2
            result = update_ui_with_state(new_state)
            result[status] = f"Model 1 done. Now generating with {model2_name}..."
            result[shared_state] = new_state
            yield result
            
            # --- Generate with Model 2 ---
            text_so_far = ""
            tokens_so_far = 0
            
            for partial_text, partial_tokens, elapsed in ollama_api.generate_streaming(
                model2_name, 
                user_prompt, 
                system_prompt=sys_prompt, 
                temperature=temp, 
                max_tokens=tokens,
                timeout=timeout_sec
            ):
                text_so_far = partial_text
                tokens_so_far = partial_tokens
                new_state["model2_text"] = text_so_far
                new_state["model2_tokens"] = tokens_so_far
                new_state["model2_time"] = elapsed
                
                result = update_ui_with_state(new_state)
                result[shared_state] = new_state
                result[status] = f"Generating response from {model2_name}..."
                yield result
            
            # 3) Done
            final_result = update_ui_with_state(new_state)
            final_result[status] = "Completed generating responses."
            final_result[shared_state] = new_state
            yield final_result
        
        # Set up event handlers
        refresh_btn.click(
            update_model_list,
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
        
        # Custom CSS injection (do not return anything here!)
        app.load(
            lambda: None,
            js="""
            () => {
                const style = document.createElement('style');
                style.textContent = `
                    .model-output {
                        min-height: 500px;
                        border: 1px solid #333;
                        border-radius: 8px;
                        padding: 16px;
                        overflow-y: auto;
                        /* Pick a dark background: */
                        background-color: #1E1E1E; 
                        /* And set the text color to something visible on dark: */
                        color: #FFFFFF;
                    }
                `;
                document.head.appendChild(style);
            }
            """
        )

        
    return app

if __name__ == "__main__":
    app = create_app()
    # For a public shareable link, add share=True (and remove debug=True if you want)
    app.launch(debug=True)
