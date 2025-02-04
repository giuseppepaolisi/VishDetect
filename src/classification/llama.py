import torch
from transformers import pipeline
import pandas as pd
from typing import List, Tuple
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from tqdm import tqdm
import csv
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

# Autenticazione Hugging Face
token = os.getenv("TOKEN_HUGGINGFACE")
login(token=token)

# Modello specifico
MODEL_LLAMA_1B_INSTRUCT = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_LLAMA_3B_INSTRUCT = "meta-llama/Llama-3.2-3B-Instruct"

class Llama1BInstructClassifier:
    """Classificatore specifico per il modello Llama-3.2-1B-Instruct"""

    def __init__(self, model_name: str, max_length: int = 2048, device: str = "cpu"):
        self.model_name = model_name
        self.max_length = max_length
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

    def _truncate_tokens(self, messages: List[dict]) -> List[dict]:
        """Tronca mantenendo i messaggi di sistema e le istruzioni"""
        # Separa messaggi essenziali (sistema/istruzioni) e conversazione
        essential_indices = [0, 1]  # Sistema e prima istruzione
        conversation_index = next((i for i, m in enumerate(messages) if "Conversation: \"" in m['content']), -1)

        essential = messages[:conversation_index]
        conversation = messages[conversation_index:] if conversation_index != -1 else []

        # Calcola token per parte essenziale
        essential_tokens = sum(len(m['content'].split()) for m in essential)

        # Tronca la conversazione se necessario
        remaining = self.max_length - essential_tokens
        if remaining <= 0:
            return self._truncate_essential(essential)

        truncated_conversation = []
        total = 0
        for msg in conversation:
            tokens = len(msg['content'].split())
            if total + tokens > remaining:
                allowed = remaining - total
                truncated_content = ' '.join(msg['content'].split()[:allowed])
                truncated_conversation.append({'role': msg['role'], 'content': truncated_content})
                break
            else:
                truncated_conversation.append(msg)
                total += tokens

        return essential + truncated_conversation

    def _truncate_essential(self, essential: List[dict]) -> List[dict]:
        """Tronca i messaggi essenziali in caso di overflow estremo"""
        total = 0
        truncated = []
        for msg in essential:
            tokens = len(msg['content'].split())
            if total + tokens > self.max_length:
                allowed = self.max_length - total
                truncated_content = ' '.join(msg['content'].split()[:allowed])
                truncated.append({'role': msg['role'], 'content': truncated_content})
                break
            else:
                truncated.append(msg)
                total += tokens
        return truncated

    def create_message(self, text: str, strategy: int) -> List[dict]:
        """Create a message formatted for Llama-3.2-1B-Instruct"""
        system_content = """Act as a vishing detection system. Analyze conversations using these prioritized criteria:

    1. **Urgency & Threats** (30% weight):
    - Immediate action demanded? (e.g., "within 24 hours")
    - Legal/financial consequences threatened?

    2. **Identity Verification** (25% weight):
    - Claims unverifiable authority? (gov, bank, tech)
    - Uses generic titles? ("Security Department")

    3. **Data Requests** (20% weight):
    - Asks for sensitive info? (passwords, SSN, PIN)
    - Requests unusual verification?

    4. **Payment Methods** (15% weight):
    - Suggests gift cards/crypto?
    - Mentions unorthodox transfer channels?

    5. **Consistency Check** (10% weight):
    - Contains logical inconsistencies?
    - Has grammatical/spelling errors?

    **Classification Rules**:
    - ≥2 strong indicators → VISHING
    - 1 weak indicator → Review needed
    - No indicators → LEGITIMATE

    **Output Format**:
    1. Indicator Analysis: [Brief factors]
    2. Confidence: [0-100%]
    3. Classification: [VISHING/LEGITIMATE]"""

        examples = [
            {
                "input": "This is Microsoft Support. Your computer has viruses. Immediately install this update: http://bit.ly/xyZ123",
                "analysis": "1. Urgency ('Immediately'), Unverified identity, Suspicious link 2. Confidence: 92% 3. VISHING"
            },
            {
                "input": "Hello, this is Chase Fraud Prevention. We detected a $458 charge in Texas. Reply YES to confirm or NO to decline.",
                "analysis": "1. Verified entity, Standard verification 2. Confidence: 85% 3. LEGITIMATE"
            }
        ]

        messages = []
        if strategy == 1:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Analyze:\n{text}"}
            ]
        elif strategy == 2:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Examples:\n" + "\n\n".join(
                    f"Input: {ex['input']}\nAnalysis: {ex['analysis']}" for ex in examples
                )},
                {"role": "user", "content": f"New Analysis:\n{text}"}
            ]
        else:
            raise ValueError("Invalid strategy. Use 1 or 2.")

        return self._truncate_tokens(messages)

    def summarize_conversation(self, text: str) -> str:
        """Genera una sintesi della conversazione"""
        messages = [
            {"role": "user", "content": f"Summarize the following conversation in a single sentence: {text}"},
        ]
        prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
        outputs = self.pipe(prompt, max_new_tokens=100, pad_token_id=self.pipe.tokenizer.eos_token_id)

        summary = outputs[0]["generated_text"].strip()
        return summary

    def classify_single(self, text: str, strategy: int, summarized: bool = False) -> Tuple[int, float]:
        """Versione corretta con formattazione del prompt"""
        if summarized:
            text = self.summarize_conversation(text)

        messages = self.create_message(text, strategy)

        # Applica il chat template del modello
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generazione con formato corretto
        outputs = self.pipe(
            prompt,
            max_new_tokens=50,
            return_full_text=False,
            temperature=0.5,
            pad_token_id=self.pipe.tokenizer.eos_token_id
        )

        response = outputs[0]['generated_text'].strip()
        return self._process_response(response)

    def _process_response(self, response: str) -> Tuple[int, float]:
        """Elabora la risposta del modello per ottenere la predizione e la confidenza."""
        response = response.strip().upper()
        base_confidence = 0.8

        if "VISHING" in response:
            return 1, base_confidence
        elif "LEGITIMATE" in response:
            return 0, base_confidence
        elif "FRAUD" in response or "SCAM" in response:
            return 1, base_confidence * 0.75
        else:
            return (1, base_confidence * 0.75) if "FRAUD" in response or "SCAM" in response else (0, base_confidence * 0.75)

    def classify_conversations(self, conversations: List[str], strategy: int, summarized: bool = False) -> Tuple[List[int], List[float]]:
        """Classifica una lista di conversazioni."""
        predictions = []
        confidences = []

        for text in tqdm(conversations, desc="Classificazione conversazioni", unit="conversazione"):
            pred, conf = self.classify_single(text, strategy, summarized)
            predictions.append(pred)
            confidences.append(conf)

        return predictions, confidences

def main(model_name: str):
    """
    Esegue la classificazione usando il modello Llama-3.2-1B-Instruct.

    Args:
        model_name (str): Nome del modello da utilizzare.
    """
    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(script_dir, '../../datasets/dataset2.1.csv')
    df = pd.read_csv(
        dataset_path,
        quoting=csv.QUOTE_NONNUMERIC,
        escapechar='\\',
        na_filter=False
    )

    # Verifica che tutti i transcript siano stringhe
    df['Transcript'] = df['Transcript'].astype(str)

    # Inizializza il classificatore
    classifier = Llama1BInstructClassifier(model_name)

    print(f"\nUtilizzo del modello: {model_name}")

    # Esegui la classificazione con ogni strategia
    for strategy in range(1, 3):
        for summarized in [False, True]:
            mode = "sintetizzata" if summarized else "originale"
            print(f"\nStrategia {strategy} con conversazione {mode}")
            predictions, confidences = classifier.classify_conversations(df['Transcript'].tolist(), strategy, summarized)

            # Crea un nuovo DataFrame con i risultati
            results_df = pd.DataFrame({
                'Transcript': df['Transcript'],
                'Label': df['Label'],
                'Predicted': predictions,
                'Confidence': confidences
            })

            # Stampa i risultati
            print("\nRapporto di classificazione:")
            print(classification_report(df['Label'], predictions))

            print("\nMatrice di confusione:")
            print(confusion_matrix(df['Label'], predictions))

            # Crea il nome del file di output con timestamp, strategia e modalità
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short_name = model_name.split('/')[-1]
            output_file = f'predictions_{model_short_name}_strategy{strategy}_{"summarized" if summarized else "original"}_{timestamp}.csv'

            # Salva il DataFrame in CSV
            results_df.to_csv(
                output_file,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                escapechar='\\',
                na_rep=''
            )

            print(f"\nRisultati salvati in: {output_file}")

if __name__ == "__main__":
    main(MODEL_LLAMA_1B_INSTRUCT)