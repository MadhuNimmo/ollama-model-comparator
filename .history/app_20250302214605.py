#!/usr/bin/env python3
"""
Ollama Model Comparator - Compare responses from different Ollama models
"""

import gradio as gr
import requests
import time
import json
import re
from difflib import SequenceMatcher

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
    
    def generate(self, model, prompt, system_prompt="", temperature=0.7, max_tokens=1000, timeout=180):
        """Generate a response from a specific model without streaming"""
        start_time = time.time()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False  # No streaming for simplicity
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout)
            end_time = time.time()
            
            if response.status_code == 200:
                response_data = response.json()
                elapsed_time = end_time - start_time
                text_response = response_data.get("response", "")
                return {
                    "response": text_response,
                    "elapsed_time": round(elapsed_time, 2),
                    "tokens": len(text_response.split()),
                    "chars": len(text_response)
                }
            else:
                return {
                    "response": f"Error: HTTP {response.status_code} - {response.text}",
                    "elapsed_time": 0,
                    "tokens": 0,
                    "chars": 0
                }
                
        except requests.exceptions.Timeout:
            return {
                "response": f"Error: Request timed out after {timeout} seconds. The model might be too large or unavailable.",
                "elapsed_time": 0,
                "tokens": 0,
                "chars": 0
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "elapsed_time": 0,
                "tokens": 0,
                "chars": 0
            }

def analyze_responses(model1_name, model2_name, model1_result, model2_result):
    """Analyze and compare the two model responses"""
    text1 = model1_result["response"]
    text2 = model2_result["response"]
    
    # Calculate similarity ratio
    similarity = SequenceMatcher(None, text1, text2).ratio()
    similarity_percent = round(similarity * 100, 1)
    
    # Basic statistics
    word_count1 = model1_result["tokens"]
    word_count2 = model2_result["tokens"]
    char_count1 = model1_result["chars"]
    char_count2 = model1_result["chars"]
    
    # Calculate word density (average word length)
    avg_word_len1 = round(char_count1 / max(word_count1, 1), 1)
    avg_word_len2 = round(char_count2 / max(word_count2, 1), 1)
    
    # Calculate response time
    time1 = model1_result["elapsed_time"]
    time2 = model2_result["elapsed_time"]
    
    # Calculate tokens per second
    tokens_per_sec1 = round(word_count1 / max(time1, 0.01), 1)
    tokens_per_sec2 = round(word_count2 / max(time2, 0.01), 1)
    
    # Estimate code blocks (simple heuristic)
    code_blocks1 = len(re.findall(r'```', text1)) // 2
    code_blocks2 = len(re.findall(r'```', text2)) // 2
    
    # Calculate different sections (paragraphs)
    paragraphs1 = len([p for p in text1.split("\n\n") if p.strip()])
    paragraphs2 = len([p for p in text2.split("\n\n") if p.strip()])
    
    # Formatting analysis
    bullets1 = len(re.findall(r'^\s*[-*+]\s', text1, re.MULTILINE))
    bullets2 = len(re.findall(r'^\s*[-*+]\s', text2, re.MULTILINE))
    
    numbered_lists1 = len(re.findall(r'^\s*\d+\.', text1, re.MULTILINE))
    numbered_lists2 = len(re.findall(r'^\s*\d+\.', text2, re.MULTILINE))
    
    # Sentiment indicators (very basic)
    negative_words = ['no', 'not', 'never', 'cannot', 'error', 'fail', 'issue', 'problem', 'difficult', 'impossible']
    positive_words = ['yes', 'can', 'success', 'solve', 'easy', 'simple', 'good', 'great', 'excellent', 'perfect']
    
    negative_count1 = sum(1 for word in negative_words if word.lower() in text1.lower())
    negative_count2 = sum(1 for word in negative_words if word.lower() in text2.lower())
    
    positive_count1 = sum(1 for word in positive_words if word.lower() in text1.lower())
    positive_count2 = sum(1 for word in positive_words if word.lower() in text2.lower())
    
    # Generate the comparison report
    comparison = f"""## Response Comparison Analysis

### Overall Similarity
- Responses are **{similarity_percent}%** similar

### Length Comparison
- **{model1_name}**: {word_count1} words, {char_count1} characters
- **{model2_name}**: {word_count2} words, {char_count2} characters
- **Difference**: {abs(word_count1 - word_count2)} words ({round(abs(word_count1 - word_count2) / max(word_count1, word_count2) * 100, 1)}%)

### Performance
- **{model1_name}**: {time1}s ({tokens_per_sec1} tokens/sec)
- **{model2_name}**: {time2}s ({tokens_per_sec2} tokens/sec)
- **Speed ratio**: {round(max(time1, 0.01) / max(time2, 0.01), 1)}x ({model1_name} vs {model2_name})

### Structure
- **{model1_name}**: {paragraphs1} paragraphs, {code_blocks1} code blocks, {bullets1} bullet points, {numbered_lists1} numbered items
- **{model2_name}**: {paragraphs2} paragraphs, {code_blocks2} code blocks, {bullets2} bullet points, {numbered_lists2} numbered items

### Style
- **{model1_name}**: Average word length: {avg_word_len1} chars
- **{model2_name}**: Average word length: {avg_word_len2} chars

### Sentiment Indicators
- **{model1_name}**: {positive_count1} positive terms, {negative_count1} negative terms
- **{model2_name}**: {positive_count2} positive terms, {negative_count2} negative terms

### Key Differences
"""
    
    # Find unique content in each response (simplified approach)
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    unique_words1 = words1 - words2
    unique_words2 = words2 - words1
    
    if len(unique_words1) > 0:
        comparison += f"- **Only in {model1_name}**: " + ", ".join(sorted(list(unique_words1)[:10]))
        if len(unique_words1) > 10:
            comparison += f" (and {len(unique_words1) - 10} more)"
        comparison += "\n"
        
    if len(unique_words2) > 0:
        comparison += f"- **Only in {model2_name}**: " + ", ".join(sorted(list(unique_words2)[:10]))
        if len(unique_words2) > 10:
            comparison += f" (and {len(unique_words2) - 10} more)"
        comparison += "\n"
    
    # Summary
    comparison += "\n### Summary\n"
    if similarity_percent > 90:
        comparison += f"The responses are very similar ({similarity_percent}% match), with minor differences in wording and structure."
    elif similarity_percent > 70:
        comparison += f"The responses are moderately similar ({similarity_percent}% match), with some differences in content and presentation."
    elif similarity_percent > 50:
        comparison += f"The responses are somewhat similar ({similarity_percent}% match), but differ significantly in content and approach."
    else:
        comparison += f"The responses are largely different ({similarity_percent}% match), taking distinct approaches to the prompt."
    
    # Performance note
    if abs(time1 - time2) > 1.0:
        faster_model = model1_name if time1 < time2 else model2_name
        comparison += f"\n\n**{faster_model}** generated its response significantly faster."
    
    # Style note
    if abs(bullets1 + numbered_lists1 - bullets2 - numbered_lists2) > 3:
        more_structured = model1_name if (bullets1 + numbered_lists1) > (bullets2 + numbered_lists2) else model2_name
        comparison += f"\n\n**{more_structured}** uses more structured formatting (lists and bullets)."
    
    # Content richness note
    if abs(word_count1 - word_count2) > 50:
        wordier_model = model1_name if word_count1 > word_count2 else model2_name
        comparison += f"\n\n**{wordier_model}** provided a more detailed response."
        
    # Code example note
    if abs(code_blocks1 - code_blocks2) > 0:
        more_code = model1_name if code_blocks1 > code_blocks2 else model2_name
        comparison += f"\n\n**{more_code}** provided more code examples."
    
    return comparison

def create_app():
    """Create and configure the Gradio app"""
    ollama_api = OllamaAPI()
    available_models = ollama_api.get_available_models()
    
    with gr.Blocks(title="Ollama Model Comparator") as app:
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
        
        # Responses area
        with gr.Tab("Side by Side"):
            with gr.Row():
                # Left model output
                with gr.Column():
                    model1_info = gr.Markdown("### Model 1")
                    model1_stats = gr.Markdown("Time: 0.00s | Tokens: 0")
                    model1_output = gr.Textbox(
                        label="Output",
                        lines=20,
                        max_lines=30,
                        show_copy_button=True
                    )
                
                # Right model output
                with gr.Column():
                    model2_info = gr.Markdown("### Model 2")
                    model2_stats = gr.Markdown("Time: 0.00s | Tokens: 0")
                    model2_output = gr.Textbox(
                        label="Output",
                        lines=20,
                        max_lines=30,
                        show_copy_button=True
                    )
        
        # Comparison tab
        with gr.Tab("Analysis"):
            comparison_output = gr.Markdown("Run a comparison to see analysis")
        
        status = gr.Markdown("Ready to compare models")
        
        # Add custom CSS
        gr.HTML("""
        <style>
        .gradio-container {
            max-width: 1200px !important;
        }
        </style>
        """)
        
        # Basic functions
        def refresh_model_list():
            """Refresh the list of available models"""
            models = ollama_api.get_available_models()
            
            if not models:
                status_msg = "No models found. Is Ollama running? Try starting it with 'ollama serve'"
            else:
                status_msg = f"Found {len(models)} models"
                
            return models, models, status_msg
            
        def add_custom_models(custom_text):
            """Add custom models to the dropdown lists"""
            if not custom_text:
                return available_models, available_models, ""
                
            custom_models = [m.strip() for m in custom_text.split(",") if m.strip()]
            
            if not custom_models:
                return available_models, available_models, ""
                
            # Combine with existing models
            existing_models = ollama_api.get_available_models()
            all_models = existing_models.copy()
            
            for m in custom_models:
                if m not in all_models:
                    all_models.append(m)
            
            return all_models, all_models, f"Added custom models: {', '.join(custom_models)}"
            
        def clear_outputs():
            """Clear the output areas"""
            return "### Model 1", "### Model 2", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Run a comparison to see analysis", "Ready to compare models"
        
        def compare_models(model1_name, model2_name, sys_prompt, user_prompt, temp, max_tok, timeout_sec):
            """Compare two models (non-streaming)"""
            if not model1_name or not model2_name:
                return "### Model 1", "### Model 2", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Run a comparison to see analysis", "Please select both models"
            
            if not user_prompt:
                return "### Model 1", "### Model 2", "Time: 0.00s | Tokens: 0", "Time: 0.00s | Tokens: 0", "", "", "Run a comparison to see analysis", "Please enter a prompt"
            
            # Update status
            status_msg = "Generating responses... This may take some time."
            
            # Sequential requests without streaming updates
            model1_result = ollama_api.generate(model1_name, user_prompt, sys_prompt, temp, max_tok, timeout_sec)
            model2_result = ollama_api.generate(model2_name, user_prompt, sys_prompt, temp, max_tok, timeout_sec)
            
            # Generate the comparison analysis
            comparison = analyze_responses(model1_name, model2_name, model1_result, model2_result)
            
            # Prepare the returns
            model1_header = f"### {model1_name}"
            model2_header = f"### {model2_name}"
            model1_status = f"Time: {model1_result['elapsed_time']:.2f}s | Tokens: {model1_result['tokens']}"
            model2_status = f"Time: {model2_result['elapsed_time']:.2f}s | Tokens: {model2_result['tokens']}"
            
            status_msg = "Comparison complete."
            
            return model1_header, model2_header, model1_status, model2_status, model1_result['response'], model2_result['response'], comparison, status_msg
        
        # Set up event handlers
        refresh_btn.click(
            fn=refresh_model_list,
            outputs=[model1, model2, status]
        )
        
        custom_models.change(
            fn=add_custom_models,
            inputs=[custom_models],
            outputs=[model1, model2, status]
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
                comparison_output,
                status
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
                comparison_output,
                status
            ]
        )
        
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()