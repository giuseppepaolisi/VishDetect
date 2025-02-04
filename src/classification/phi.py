import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd
import csv
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

# Autenticazione Hugging Face
token = os.getenv("TOKEN_HUGGINGFACE")
login(token=token)

# Modello specifico
MODEL_PHI_15 = "microsoft/phi-1_5"
MODEL_PHI_2 = "microsoft/phi-2"
MODEL_PHI_3 = "microsoft/Phi-3-mini-128k-instruct"

default_device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(default_device)

class PhiClassifier:
    """Classificatore adattato per phi"""

    def __init__(self, model_name: str, max_length: int = 2048):
        self.model_name = model_name
        self.max_length = max_length
        self.device = default_device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

    def create_message(self, text: str, strategy: int) -> str:
            """Format prompt according to DeepSeek-R1 specifications"""
            system_content = (
                "You are a cybersecurity expert specialized in detecting vishing (voice phishing) attacks. "
                "Analyze conversations using these key indicators:\n\n"
                "1. Caller Identity: Claims to represent official entities (e.g., bank, government agency)\n"
                "2. Urgency Tactics: Uses time pressure phrases (e.g., 'immediately', 'account suspension')\n"
                "3. Suspicious Requests: Asks for sensitive data or financial transactions\n"
                "4. Threat Elements: Contains threats of legal action or consequences\n"
                "5. Context Consistency: Logical flaws or unrealistic official references\n\n"
                "Classification Rules:\n"
                "- Perform step-by-step analysis of all indicators\n"
                "- Require at least TWO strong indicators for VISHING classification\n"
                "- LEGITIMATE conversations show normal communication patterns\n"
                "- Consider cultural context and regional institutional procedures"
            )

            example_template = (
                "=== Positive Example (VISHING) ===\n"
                "Conversation: 'This is Officer Smith from IRS. Your social security number is linked to drug trafficking. "
                "Wire $5,000 immediately to case number XBZ-456 to avoid arrest.'\n"
                "Analysis:\n"
                "1. Identity: Claims government authority (IRS)\n"
                "2. Urgency: 'Immediately' pressure\n"
                "3. Request: Wire transfer demand\n"
                "4. Threat: Arrest mention\n"
                "5. Inconsistency: IRS doesn't demand wire transfers\n"
                "Classification: VISHING\n\n"
                "=== Negative Example (LEGITIMATE) ===\n"
                "Conversation: 'Hello Ms. Johnson, this is Amazon security team. We detected unusual login from Italy. "
                "Did you authorize this? Please verify your account via our official app.'\n"
                "Analysis:\n"
                "1. Identity: Legitimate service provider\n"
                "2. Urgency: Appropriate security alert\n"
                "3. Request: Standard verification process\n"
                "4. Threat: None present\n"
                "5. Consistency: Matches standard protocols\n"
                "Classification: LEGITIMATE\n\n"
            )

            if strategy == 1:
                prompt = (
                    f"{system_content}\n\n"
                    f"Analyze this conversation:\n«{text}»\n\n"
                    "Step-by-step evaluation:\n1."
                )
            elif strategy == 2:
                prompt = (
                    f"{system_content}\n\n{example_template}"
                    f"New conversation to analyze:\n«{text}»\n\n"
                    "Step-by-step evaluation:\n1."
                )
            else:
                raise ValueError("Invalid strategy. Use 1 or 2.")

            return prompt

    def classify_single(self, text: str, strategy: int, summarized: bool = False) -> Tuple[int, float]:
        """Classify a single conversation."""
        if summarized :
          text = self.summarize_conversation(text)
        prompt = self.create_message(text, strategy)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.6,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._process_response(response)

    def _process_response(self, response: str) -> Tuple[int, float]:
        """Process the model's response to extract the prediction and confidence."""
        response = response.strip().upper()
        base_confidence = 0.8

        if "VISHING" in response:
            return 1, base_confidence
        elif "LEGITIMATE" in response or "NOT VISHING" in response:
            return 0, base_confidence
        else:
            return 0, base_confidence * 0.75

    def classify_conversations(self, conversations: List[str], strategy: int) -> Tuple[List[int], List[float]]:
        """Classify a list of conversations."""
        predictions = []
        confidences = []

        for text in tqdm(conversations, desc="Classifying conversations", unit="conversation"):
            pred, conf = self.classify_single(text, strategy)
            predictions.append(pred)
            confidences.append(conf)

        return predictions, confidences


    def summarize_conversation(self, text: str) -> str:
        """Genera una sintesi della conversazione"""
        prompt = f"Summarize this conversation in one concise sentence:\n{text}\nSummary:"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=self.max_length, truncation=True
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.6,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        summary = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return summary.strip()

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
    classifier = PhiClassifier(model_name)

    print(f"\nUtilizzo del modello: {model_name}")

    # Esegui la classificazione con ogni strategia
    for strategy in range(1, 3):
        for summarized in [False, True]:
            mode = "sintetizzata" if summarized else "originale"
            print(f"\nStrategia {strategy} con conversazione {mode}")
            # The summarized argument was removed in the following line
            predictions, confidences = classifier.classify_conversations(df['Transcript'].tolist(), strategy)

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
    main(MODEL_PHI_15)