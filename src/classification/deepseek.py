import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
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
MODEL_DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

class DeepSeekClassifier:
    """Classificatore adattato per DeepSeek-R1-Distill-Qwen-1.5B"""

    def __init__(self, model_name: str, max_length: int = 2048, device: str = "cpu"):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=device,
            temperature=0.6
        )

    def create_message(self, text: str, strategy: int) -> str:
        """Formatta il prompt secondo le specifiche DeepSeek-R1"""
        system_content = (
            "Vishing, or voice phishing, is a social engineering attack where malicious actors use phone calls to deceive individuals into revealing sensitive information, transferring money, or granting access to systems. These attacks often impersonate trusted entities (e.g., banks, government agencies, or tech support) and exploit urgency, fear, or authority to manipulate victims. Detecting vishing attempts is critical to preventing financial loss, identity theft, and data breaches. Objective: Analyze the provided written text to determine if it contains indicators of a vishing attempt."
            "Analyze if the conversation appears to be a normal, everyday discussion between two people, with no indications of deception or suspicious behavior this is LEGITIMATE."
        )

        if strategy == 1:
            prompt = f"{system_content}\nAnalize this conversation and classificate it with VISHING or LEGITIMATE. Conversation: \"{text}\"\nAnalysis:"
        elif strategy == 2:
            prompt = (
                f"{system_content}\nExample Analysis:\nConversation:"
                "\"Good afternoon, Seoul Central District Prosecutors' Office. I'm Investigator Lee Jin-ho from the Advanced Crime Investigation Team 1. I'm here to check on a few things regarding your personal information leak. Do you mind if we talk? I just wanted to ask you a few questions. Do you know a person named Ji-kyung?"
                "I’ll tell you a little bit about the computer first. We recently arrested a financial crime fraud group centered on Moon Hee-kyung in Osaka. A large number of debit card passbooks were found at the scene. The passbooks also revealed the identities of those involved. When I checked, they were issued every hour in Gwangmyeong, Gyeonggi Province. Do you have a new name?"
                "If you opened it yourself today—no, it’s a Woori Bank or Hana Bank passbook.\""
                f"\nAnalysis: VISHING\n\n"
                f"Analize this conversation and classificate it with VISHING or LEGITIMATE. Conversation: \"{text}\"\nAnalysis:"
            )
        else:
            raise ValueError("Invalid strategy. Use 1 or 2.")

        return prompt

    def classify_single(self, text: str, strategy: int, summarized: bool = False) -> Tuple[int, float]:
        """Classifica una singola conversazione"""
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

    def summarize_conversation(self, text: str) -> str:
        prompt = f"Summarize this conversation:\n{text}\nSummary:"

        outputs = self.pipe(
            prompt,
            max_new_tokens=100,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )

        return outputs[0]["generated_text"].strip()

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
    df = pd.read_csv(
        'dataset_tradotto1.3.csv',
        quoting=csv.QUOTE_NONNUMERIC,
        escapechar='\\',
        na_filter=False
    )

    # Verifica che tutti i transcript siano stringhe
    df['Transcript'] = df['Transcript'].astype(str)

    # Inizializza il classificatore
    classifier = DeepSeekClassifier(model_name)

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
    main(MODEL_DEEPSEEK)