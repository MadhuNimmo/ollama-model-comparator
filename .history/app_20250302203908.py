#!/usr/bin/env python3
"""
Ollama Model Comparator - Compare responses from different Ollama models
"""

import gradio as gr
import requests
import time
import difflib
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

class OllamaAPI:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        
    def get_available_models(self):
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                return [model["name"] for model in models_data["models"]]
            else:
                return []
        except Exception as e:
            print(f"Error getting models: {e}")
            return []
            
    def generate_response(self, model, prompt, system_prompt="", temperature=0.7, max_tokens=1000):
        """Generate a response from a specific model"""
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
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                elapsed_time = time.time() - start_time
                
                return {
                    "model": model,
                    "response": result.get("response", ""),
                    "elapsed_time": round(elapsed_time, 2),
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0),
                    "eval_duration": result.get("eval_duration", 0)
                }
            else:
                return {
                    "model": model, 
                    "response": f"Error: {response.status_code} - {response.text}", 
                    "elapsed_time": 0
                }
        except Exception as e:
            return {"model": model, "response": f"Error: {str(e)}", "elapsed_time": 0}
    
    def batch_generate(self, models, prompt, system_prompt="", temperature=0.7, max_tokens=1000):
        """Generate responses from multiple models in parallel"""
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            responses = list(executor.map(
                lambda model: self.generate_response(
                    model, prompt, system_prompt, temperature, max_tokens
                ),
                models
            ))
        return responses

def compare_responses(responses):
    """Compare responses from different models"""
    if len(responses) < 2:
        return {"comparison": "Need at least two models to compare"}
    
    texts = [resp["response"] for resp in responses]
    models = [resp["model"] for resp in responses]
    
    # Calculate text similarity matrix
    similarity_matrix = {}
    for i, (model1, text1) in enumerate(zip(models, texts)):
        similarity_matrix[model1] = {}
        for j, (model2, text2) in enumerate(zip(models, texts)):
            if i != j:
                # Calculate similarity ratio using difflib
                similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
                similarity_matrix[model1][model2] = round(similarity * 100, 1)
    
    # Performance comparison
    performance = {
        "response_time": {model: resp["elapsed_time"] for model, resp in zip(models, responses)},
        "tokens_per_second": {
            model: round(resp["eval_count"] / (resp["eval_duration"] / 1e9), 2) 
            if resp.get("eval_duration", 0) > 0 else 0
            for model, resp in zip(models, responses)
        }
    }
    
    # Response length comparison
    length_comparison = {model: len(resp["response"].split()) for model, resp in zip(models, responses)}
    
    return {
        "similarity_matrix": similarity_matrix,
        "performance": performance,
        "length_comparison": length_comparison
    }

def generate_diff_html(text1, text2, model1, model2):
    """Generate HTML diff between two model responses"""
    diff = difflib.HtmlDiff()
    html = diff.make_file(text1.splitlines(), text2.splitlines(), model1, model2)
    
    # Make the HTML more compact and mobile-friendly
    html = html.replace('<table class="diff"', '<table class="diff" style="width:100%; font-size:14px;"')
    
    return html

def main():
    ollama_api = OllamaAPI()
    default_models = ollama_api.get_available_models()
    
    with gr.Blocks(title="Ollama Model Comparator", theme=gr.themes.Soft(primary_hue="blue")) as app:
        gr.Markdown(
            """
            # 🤖 Ollama Model Comparator
            
            Compare responses from different Ollama models side by side.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Ollama API URL input
                ollama_url = gr.Textbox(
                    label="Ollama API URL", 
                    value="http://localhost:11434",
                    placeholder="http://localhost:11434"
                )
                
                # Refresh models button
                refresh_btn = gr.Button("Refresh Models")
                
                # Model input for custom models not in the list
                custom_models = gr.Textbox(
                    label="Custom Models (comma-separated)", 
                    placeholder="llama3,mistral,codellama"
                )
                
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=default_models,
                    value=default_models[:2] if len(default_models) >= 2 else default_models,
                    label="Select Models to Compare",
                    multiselect=True
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
                    generate_btn = gr.Button("Generate Responses", variant="primary")
                    clear_btn = gr.Button("Clear Results")
            
            with gr.Column(scale=1):
                status = gr.Markdown("### Status: Ready")
                
                # Stats
                with gr.Accordion("Response Statistics", open=False):
                    stats_table = gr.DataFrame(
                        headers=["Model", "Words", "Time (s)", "Tokens/sec"],
                        label="Response Statistics"
                    )
                
                # Similarity matrix
                with gr.Accordion("Similarity Matrix (%)", open=False):
                    similarity_table = gr.DataFrame(label="Similarity Matrix")
        
        # Response tabs
        with gr.Tabs() as response_tabs:
            # Output boxes will be dynamically added here
            output1 = gr.Markdown(label="Model 1")
            output2 = gr.Markdown(label="Model 2")
            
            # Diff view tab
            with gr.Tab("Side-by-Side Diff"):
                diff_view = gr.HTML(label="Response Differences")
                
                with gr.Row():
                    diff_model1 = gr.Dropdown(label="Model 1")
                    diff_model2 = gr.Dropdown(label="Model 2")
                    update_diff_btn = gr.Button("Update Diff")

        # Store model responses for diff view
        model_responses = gr.State({})
        selected_models = gr.State([])
        
        def refresh_models_list(url):
            api = OllamaAPI(url)
            models = api.get_available_models()
            return gr.Dropdown(choices=models, value=models[:2] if len(models) >= 2 else models)
        
        def update_models(dropdown_models, custom_model_text):
            if not custom_model_text:
                return dropdown_models
            
            custom_models = [m.strip() for m in custom_model_text.split(",") if m.strip()]
            all_models = list(dropdown_models) + [m for m in custom_models if m not in dropdown_models]
            return all_models
        
        def generate(api_url, models, system_prompt, user_prompt, temp, tokens):
            if not models:
                return {
                    status: "### Status: ⚠️ Please select at least one model",
                    selected_models: []
                }
            
            # Update status
            yield {
                status: f"### Status: 🔄 Generating responses for {len(models)} models...",
                selected_models: models,
                # Clear previous outputs
                output1: "",
                output2: "",
                diff_view: "",
                stats_table: None,
                similarity_table: None,
                diff_model1: gr.Dropdown(choices=models, value=models[0] if models else None),
                diff_model2: gr.Dropdown(choices=models, value=models[1] if len(models) > 1 else None)
            }
            
            try:
                # Generate responses
                api = OllamaAPI(api_url)
                responses = api.batch_generate(models, user_prompt, system_prompt, temp, tokens)
                
                # Store responses for later use
                responses_dict = {resp["model"]: resp["response"] for resp in responses}
                
                # Calculate comparison metrics
                comparison = compare_responses(responses)
                
                # Prepare stats table
                stats_data = [[
                    resp["model"],
                    len(resp["response"].split()),
                    resp["elapsed_time"],
                    comparison["performance"]["tokens_per_second"].get(resp["model"], 0) 
                    if "performance" in comparison and "tokens_per_second" in comparison["performance"] else 0
                ] for resp in responses]
                
                # Prepare similarity matrix
                sim_matrix = []
                models_list = [resp["model"] for resp in responses]
                
                # Add header row
                sim_matrix.append([""] + models_list)
                
                # Add data rows
                for model1 in models_list:
                    row = [model1]
                    for model2 in models_list:
                        if model1 == model2:
                            row.append(100.0)
                        else:
                            row.append(comparison["similarity_matrix"].get(model1, {}).get(model2, 0) 
                                       if "similarity_matrix" in comparison else 0)
                    sim_matrix.append(row)
                
                # Generate diff view for first two models if available
                diff_html = ""
                if len(responses) >= 2:
                    diff_html = generate_diff_html(
                        responses[0]["response"], 
                        responses[1]["response"],
                        responses[0]["model"],
                        responses[1]["model"]
                    )
                
                # Format responses for display
                formatted_outputs = []
                for i, resp in enumerate(responses):
                    formatted_outputs.append(f"### {resp['model']} Response\n\n{resp['response']}")
                
                # Fill in missing outputs with empty strings
                while len(formatted_outputs) < 2:
                    formatted_outputs.append("")
                
                yield {
                    status: f"### Status: ✅ Generated responses for {len(models)} models",
                    model_responses: responses_dict,
                    stats_table: stats_data,
                    similarity_table: sim_matrix,
                    output1: formatted_outputs[0],
                    output2: formatted_outputs[1] if len(formatted_outputs) > 1 else "",
                    diff_view: diff_html,
                }
                
            except Exception as e:
                yield {
                    status: f"### Status: ❌ Error: {str(e)}",
                    selected_models: models
                }
        
        def clear_results():
            return {
                status: "### Status: Ready",
                output1: "",
                output2: "",
                diff_view: "",
                stats_table: None,
                similarity_table: None,
                model_responses: {}
            }
        
        def update_diff(stored_responses, model1, model2):
            if not model1 or not model2 or model1 not in stored_responses or model2 not in stored_responses:
                return "<p>Select two models with valid responses to view differences</p>"
            
            try:
                text1 = stored_responses[model1]
                text2 = stored_responses[model2]
                return generate_diff_html(text1, text2, model1, model2)
            except Exception as e:
                return f"<p>Error generating diff: {str(e)}</p>"
        
        # Connect event handlers
        refresh_btn.click(
            refresh_models_list,
            inputs=[ollama_url],
            outputs=[model_dropdown]
        )
        
        generate_btn.click(
            generate,
            inputs=[
                ollama_url,
                model_dropdown,
                system_prompt,
                prompt,
                temperature,
                max_tokens
            ],
            outputs=[
                status,
                selected_models,
                output1,
                output2,
                diff_view,
                stats_table,
                similarity_table,
                diff_model1,
                diff_model2,
                model_responses
            ]
        )
        
        clear_btn.click(
            clear_results,
            outputs=[
                status,
                output1,
                output2,
                diff_view,
                stats_table,
                similarity_table,
                model_responses
            ]
        )
        
        update_diff_btn.click(
            update_diff,
            inputs=[model_responses, diff_model1, diff_model2],
            outputs=[diff_view]
        )
        
        # Add custom models to dropdown selections
        custom_models.change(
            update_models,
            inputs=[model_dropdown, custom_models],
            outputs=[model_dropdown]
        )
    
    app.launch()

if __name__ == "__main__":
    main()