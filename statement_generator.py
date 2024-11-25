import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize model and tokenizer
print("Initializing models...", file=sys.stderr)
generator_model_name = "Qwen/Qwen2-1.5B"
generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
generator_model = AutoModelForCausalLM.from_pretrained(generator_model_name)

def generate_statement(input_text, input_label, mode):
   try:
       if mode == 'res':
           prompt = f"""Generate a diplomatic response to this message:
Text: {input_text}
Type: {input_label}
Requirements:
- Formal diplomatic tone
- Address specific points
- No country names
- Clear and focused
- 2-3 sentences only

Response:"""
       else:
           prompt = f"""Provide diplomatic recommendations for this message:
Text: {input_text}  
Type: {input_label}
Provide:
1. Response strategy
2. Key points to address
3. Risk assessment
4. Next steps
No country names or politics.

Recommendations:"""

       messages = [
           {"role": "system", "content": "You are a diplomatic communication expert."},
           {"role": "user", "content": prompt}
       ]
       
       encoded = generator_tokenizer.apply_chat_template(
           messages,
           return_tensors="pt",
           add_generation_prompt=True
       )
       
       outputs = generator_model.generate(
           encoded,
           max_new_tokens=500 if mode=='rec' else 250,
           temperature=0.8 if mode=='rec' else 0.7,
           top_p=0.9,
           do_sample=True,
           repetition_penalty=1.2
       )
       
       response = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
       
       if mode == 'res':
           content = response.split("Response:")[-1].strip()
           output_key = "response"
       else:
           content = response.split("Recommendations:")[-1].strip()
           output_key = "recommendation"
           
       return output_key, content

   except Exception as e:
       print(f"Error: {e}", file=sys.stderr)
       return ("response" if mode=='res' else "recommendation"), "Error generating content"

def main():
   for line in sys.stdin:
       try:
           data = json.loads(line.strip())
           output_key, content = generate_statement(data['text'], data['label'], 'res')
           
           result = {
               "text": data['text'],
               "label": data['label'], 
               output_key: content
           }
           
           print(json.dumps(result, ensure_ascii=False))
           sys.stdout.flush()
           
       except Exception as e:
           print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
   main()