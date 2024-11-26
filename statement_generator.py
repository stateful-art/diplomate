import sys
import json
import torch
from datetime import datetime
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Generate diplomatic responses or recommendations')
    parser.add_argument('--mode', type=str, choices=['res', 'rec'], required=False,
                       help='Mode: "res" for direct response, "rec" for recommendations. If not specified, generates both.')
    return parser.parse_args()

# Initialize model and tokenizer
print("Initializing models...", file=sys.stderr)
generator_model_name = "Qwen/Qwen2-1.5B"
generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
generator_model = AutoModelForCausalLM.from_pretrained(generator_model_name)

def get_response_contexts():
    return {
        "cooperation": "Foster mutual benefit while maintaining clear operational boundaries.",
        "negotiation": "Engage in constructive dialogue while preserving core interests.",
        "alliance_proposal": "Show measured interest in cooperation while preserving independent decision-making.",
        "threat": "Maintain firm positioning while emphasizing diplomatic solutions.",
        "intimidation": "Demonstrate unwavering resolve while keeping dialogue channels open.",
        "compromise": "Acknowledge mutual interests while ensuring balanced concessions.",
        "peace_offer": "Consider peace initiatives while maintaining prudent deliberation.",
        "declaration_of_war": "Maintain composure while asserting defensive readiness.",
        "ceasefire_request": "Address humanitarian concerns while ensuring security parameters.",
        "trade_proposal": "Evaluate economic opportunities while ensuring mutual benefit.",
        "intelligence_sharing": "Consider security cooperation while maintaining operational discretion.",
        "diplomatic_pressure": "Address concerns while maintaining diplomatic dignity.",
        "sanctions_threat": "Respond to concerns while emphasizing diplomatic alternatives.",
        "mediation_offer": "Consider third-party facilitation while maintaining sovereignty.",
        "neutrality_declaration": "Affirm non-intervention while maintaining diplomatic relations.",
        "territorial_claim": "Assert territorial integrity while remaining open to dialogue.",
        "diplomatic_protest": "Address grievances while maintaining professional composure.",
        "apology": "Express appropriate regret while maintaining diplomatic dignity.",
        "praise_or_commendation": "Acknowledge achievements while maintaining professional distance.",
        "criticism": "Address concerns while maintaining diplomatic discourse.",
        "request_for_aid": "Consider assistance needs while following proper protocols.",
        "offer_of_assistance": "Express support while establishing appropriate frameworks.",
        "ultimatum": "Maintain resolve while preserving diplomatic options.",
        "non_aggression_pact": "Consider security assurances while maintaining sovereignty.",
        "treaty_proposal": "Evaluate cooperative frameworks while ensuring national interests.",
        "diplomatic_recognition": "Acknowledge diplomatic status while following proper procedures.",
        "severance_of_relations": "Maintain dignity while following diplomatic protocols.",
        "espionage_accusation": "Address security concerns while maintaining diplomatic channels.",
        "denial_of_accusations": "Present position clearly while maintaining professional tone.",
        "call_for_unity": "Consider collective action while maintaining autonomous decision-making.",
        "appeal_to_international_law": "Reference legal frameworks while maintaining diplomatic discourse.",
        "economic_cooperation": "Explore mutual benefits while maintaining regulatory autonomy.",
        "cultural_exchange": "Promote cultural understanding while following diplomatic protocols.",
        "military_cooperation": "Consider security collaboration while maintaining operational independence.",
        "humanitarian_aid_offer": "Coordinate assistance while ensuring proper procedures.",
        "request_for_mediation": "Consider conflict resolution while maintaining sovereign rights.",
        "diplomatic_immunity_invocation": "Assert diplomatic privileges while maintaining professional conduct.",
        "extradition_request": "Process legal matters through appropriate diplomatic channels.",
        "asylum_offer": "Handle humanitarian matters through established protocols.",
        "propaganda": "Address information concerns while maintaining diplomatic composure.",
        "disinformation": "Counter misrepresentation while maintaining professional standards.",
        "confidence_building_measure": "Foster trust while maintaining appropriate boundaries.",
        "arms_control_proposal": "Consider security measures while maintaining defense capabilities.",
        "environmental_cooperation": "Promote ecological collaboration while ensuring sovereign interests.",
        "technology_transfer": "Facilitate technical exchange within appropriate frameworks.",
        "diplomatic_demarche": "Convey position firmly while maintaining diplomatic protocol.",
        "formal_complaint": "Address grievances through proper diplomatic channels.",
        "request_for_clarification": "Seek information while maintaining professional discourse.",
        "expression_of_concern": "Voice concerns while maintaining diplomatic engagement.",
        "congratulatory_message": "Express recognition while maintaining professional tone.",
        "condolences": "Express sympathy while maintaining diplomatic propriety.",
        "neutral_statement": "Maintain balanced position while ensuring clear communication.",
        "procedural_communication": "Follow diplomatic protocols while ensuring clear transmission.",
        "information_request": "Seek details through appropriate diplomatic channels.",
        "summit_proposal": "Consider high-level dialogue while maintaining proper preparation.",
        "arbitration_request": "Consider dispute resolution while following established procedures.",
        "border_dispute_resolution": "Address territorial matters through diplomatic channels.",
        "diplomatic_crisis_management": "Handle urgent matters while maintaining diplomatic protocol.",
        "economic_sanctions_announcement": "Implement measures while maintaining diplomatic channels.",
        "humanitarian_corridor_request": "Address humanitarian needs while ensuring security protocols.",
        "peacekeeping_mission_proposal": "Consider stability operations while maintaining sovereignty.",
        "condemnation": "Express strong disapproval while maintaining diplomatic language."
    }

def get_recommendation_contexts():
    return {
        "cooperation": "Analyze cooperation potential and framework requirements.",
        "negotiation": "Evaluate negotiation positions and potential compromises.",
        "alliance_proposal": "Assess strategic implications and commitment requirements.",
        "threat": "Analyze threat credibility and response options.",
        "intimidation": "Evaluate power dynamics and strategic responses.",
        "compromise": "Assess concession balance and strategic implications.",
        "peace_offer": "Evaluate peace terms and implementation requirements.",
        "declaration_of_war": "Analyze conflict escalation and diplomatic options.",
        "ceasefire_request": "Assess security implications and verification needs.",
        "trade_proposal": "Evaluate economic benefits and regulatory requirements.",
        "intelligence_sharing": "Assess information value and security protocols.",
        "diplomatic_pressure": "Analyze leverage points and response strategies.",
        "sanctions_threat": "Evaluate economic impact and mitigation options.",
        "mediation_offer": "Assess mediator neutrality and process framework.",
        "neutrality_declaration": "Evaluate implications and verification measures.",
        "territorial_claim": "Analyze legal basis and strategic implications.",
        "diplomatic_protest": "Assess grievance validity and response options.",
        "apology": "Evaluate appropriate response and future implications.",
        "praise_or_commendation": "Consider reciprocation and relationship building.",
        "criticism": "Analyze validity and response strategy.",
        "request_for_aid": "Assess needs and response capabilities.",
        "offer_of_assistance": "Evaluate aid implications and coordination needs.",
        "ultimatum": "Analyze demands and response options.",
        "non_aggression_pact": "Evaluate security implications and verification needs.",
        "treaty_proposal": "Assess terms and implementation requirements.",
        "diplomatic_recognition": "Evaluate implications and procedural requirements.",
        "severance_of_relations": "Analyze impact and contingency measures.",
        "espionage_accusation": "Assess evidence and response strategy.",
        "denial_of_accusations": "Evaluate defense strategy and evidence presentation.",
        "call_for_unity": "Assess collective action implications.",
        "appeal_to_international_law": "Evaluate legal basis and precedents.",
        "economic_cooperation": "Analyze economic benefits and risks.",
        "cultural_exchange": "Evaluate cultural impact and program requirements.",
        "military_cooperation": "Assess security benefits and operational protocols.",
        "humanitarian_aid_offer": "Evaluate aid coordination and distribution.",
        "request_for_mediation": "Assess mediation framework and requirements.",
        "diplomatic_immunity_invocation": "Evaluate legal basis and implications.",
        "extradition_request": "Assess legal requirements and procedures.",
        "asylum_offer": "Evaluate humanitarian and security implications.",
        "propaganda": "Analyze messaging impact and response strategy.",
        "disinformation": "Assess information integrity and counter-measures.",
        "confidence_building_measure": "Evaluate trust-building potential.",
        "arms_control_proposal": "Assess verification and compliance measures.",
        "environmental_cooperation": "Evaluate environmental impact and resources.",
        "technology_transfer": "Assess technical benefits and security implications.",
        "diplomatic_demarche": "Evaluate message impact and delivery strategy.",
        "formal_complaint": "Analyze grievance basis and response options.",
        "request_for_clarification": "Assess information needs and response strategy.",
        "expression_of_concern": "Evaluate situation gravity and response options.",
        "congratulatory_message": "Consider appropriate reciprocation.",
        "condolences": "Assess appropriate sympathy expression.",
        "neutral_statement": "Evaluate balance and positioning strategy.",
        "procedural_communication": "Assess protocol requirements.",
        "information_request": "Evaluate information sharing parameters.",
        "summit_proposal": "Assess meeting framework and preparations.",
        "arbitration_request": "Evaluate dispute resolution process.",
        "border_dispute_resolution": "Analyze territorial issues and solutions.",
        "diplomatic_crisis_management": "Assess crisis severity and response options.",
        "economic_sanctions_announcement": "Evaluate economic impact and duration.",
        "humanitarian_corridor_request": "Assess security and logistics requirements.",
        "peacekeeping_mission_proposal": "Evaluate mission scope and requirements.",
        "condemnation": "Analyze situation severity and response tone."
    }

def get_response_prompt(input_text, input_label):
    context_prompts = get_response_contexts()
    context_prompt = context_prompts.get(input_label, "Formulate a balanced diplomatic response appropriate to the situation.")

    return f"""You are a senior diplomat representing your nation. Generate ONLY the response text, without any greetings, signatures, headers or footers.

CONTEXT: {context_prompt}

INCOMING MESSAGE: {input_text}
MESSAGE TYPE: {input_label}

YOUR ROLE: Senior diplomat responding to an official communication 
TASK: Generate only the response content - no meta text, no salutations, no signatures

REQUIREMENTS:
- Generate only the response content
- No greetings, titles, or signatures
- No meta-commentary about the response
- No rephrasing of the original message
- Address the specific points directly 
- Use formal diplomatic language
- Show appropriate firmness/flexibility
- Avoid mentioning specific countries
- Keep response focused and clear

DIPLOMATIC RESPONSE:"""

def get_recommendation_prompt(input_text, input_label):
    context_prompts = get_recommendation_contexts()
    context_prompt = context_prompts.get(input_label, "Provide strategic diplomatic guidance appropriate to the situation.")

    return f"""You are a senior diplomatic advisor providing strategic guidance. Generate ONLY the recommendations content, organized in clear sections.

CONTEXT: {context_prompt}

RECEIVED MESSAGE: {input_text}
MESSAGE TYPE: {input_label}

YOUR ROLE: Senior diplomatic advisor
TASK: Generate only strategic recommendations - no meta text, no commentary

PROVIDE NUMBERED RECOMMENDATIONS ON:

1. Initial Response Strategy
   - Recommended tone and approach
   - Key points to address
   - Elements to avoid
   - Language suggestions

2. Risk Assessment
   - Potential escalation points
   - Diplomatic pitfalls to avoid
   - Implications for relations
   - Opportunities to leverage

3. Action Steps
   - Immediate actions needed
   - Follow-up measures
   - Timeline recommendations
   - Communication channels

REQUIREMENTS:
- Generate only numbered recommendations
- No greetings or signatures
- No meta-commentary
- No rephrasing of original message
- Focus on actionable guidance
- Avoid naming specific countries

DIPLOMATIC RECOMMENDATIONS:"""

def generate_diplomatic_content(input_text, input_label, mode):
    try:
        if mode == 'res':
            prompt = get_response_prompt(input_text, input_label)
            messages = [
                {"role": "system", "content": "You are a senior diplomat crafting an official response."},
                {"role": "user", "content": prompt}
            ]
        else:
            prompt = get_recommendation_prompt(input_text, input_label)
            messages = [
                {"role": "system", "content": "You are a senior diplomatic advisor providing strategic guidance."},
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
            content = response.split("DIPLOMATIC RESPONSE:")[-1].strip()
            output_key = "response"
        else:
            content = response.split("DIPLOMATIC RECOMMENDATIONS:")[-1].strip()
            output_key = "recommendation"
            
        return output_key, content

    except Exception as e:
        print(f"Error generating content: {e}", file=sys.stderr)
        return ("response" if mode=='res' else "recommendation"), f"Error in {mode} generation"

def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"qwen2-1.5b_results_{timestamp}.txt"
    
    print(f"Starting generator in {args.mode if args.mode else 'both'} mode(s)...", file=sys.stderr)
    
    for line in sys.stdin:
        try:
            input_data = json.loads(line.strip())
            results = []
            
            if args.mode in ['res', None]:
                output_key, response = generate_diplomatic_content(
                    input_data['text'], 
                    input_data['label'],
                    'res'
                )
                results.append({
                    "text": input_data['text'],
                    "label": input_data['label'],
                    output_key: response
                })
            
            if args.mode in ['rec', None]:
                output_key, recommendation = generate_diplomatic_content(
                    input_data['text'], 
                    input_data['label'],
                    'rec'
                )
                results.append({
                    "text": input_data['text'],
                    "label": input_data['label'],
                    output_key: recommendation
                })
            
            # Print to terminal and write to file
            for result in results:
                result_json = json.dumps(result, ensure_ascii=False)
                print(result_json)
                with open(output_file, 'a') as f:
                    f.write(result_json + '\n')
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error in main loop: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()