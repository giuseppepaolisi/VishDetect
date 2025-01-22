import torch
from transformers import pipeline
import pandas as pd
from typing import List, Tuple
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from tqdm import tqdm
import csv
from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()
token = os.getenv('TOKEN_HUGGINGFACE')

login(token=token)

# Modello specifico
MODEL_LLAMA_1B_INSTRUCT = "meta-llama/Llama-3.2-1B-Instruct"

class Llama1BInstructClassifier:
    """Classificatore specifico per il modello Llama-3.2-1B-Instruct"""

    def __init__(self, model_name: str, max_length: int = 2048):
        self.model_name = model_name
        self.max_length = max_length
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def truncate_text(self, text: str, prompt_template: str) -> str:
        """Tronca il testo per rispettare la lunghezza massima del modello"""
        max_text_tokens = self.max_length - len(prompt_template) - 50

        if len(text) > max_text_tokens:
            text = text[:max_text_tokens] + " [TRUNCATED]"

        return text

    def summarize_conversation(self, text: str) -> str:
        """Genera una sintesi della conversazione"""
        prompt = f"""Summarize the following conversation in a single sentence:\nConversation: {text}.\nSummary:"""

        outputs = self.pipe(prompt, max_new_tokens=50)
        summary = outputs[0]["generated_text"].strip()
        return summary

    def create_prompt(self, text: str, strategy: int) -> str:
        """Crea un prompt specifico in base alla strategia"""
        if strategy == 1:
            template = """A vishing conversation is a type of phone fraud where a scammer pretends to be a trusted entity, like a bank or government agency, to trick the victim into providing sensitive information or performing specific actions.\nAnalyze the following conversation and determine if it's a vishing attempt (phone fraud) or a legitimate conversation. Answer with 'VISHING' or 'LEGITIMATE'.\nConversation: {conversation}."""
        elif strategy == 2:
            template = """A vishing conversation is a type of phone fraud where a scammer pretends to be a trusted entity, like a bank or government agency, to trick the victim into providing sensitive information or performing specific actions.\nHere is an example of a vishing conversation:\nExample: \"A scammer, posing as Cheolmin Park from the Audit Department, repeatedly insists on resolving a \"fraudulent transaction\" involving the victim’s account. The caller uses complex and confusing language to create urgency, claiming issues with processing paperwork and requiring the victim to pay additional funds (e.g., $4 million won or $150) to resolve the matter, avoid penalties, and secure reimbursements.\"\nNow analyze the following conversation and determine if it's a vishing attempt (phone fraud) or a legitimate conversation. Answer with 'VISHING' or 'LEGITIMATE'.\nConversation: {conversation}."""
        else:
            raise ValueError("Strategia non valida. Usare 1 o 2.")

        return template.format(conversation=self.truncate_text(text, template))

    def classify_single(self, text: str, strategy: int, summarized: bool = False) -> Tuple[int, float]:
        """Classifica una singola conversazione"""
        if summarized:
            text = self.summarize_conversation(text)

        prompt = self.create_prompt(text, strategy)
        outputs = self.pipe(prompt, max_new_tokens=20, temperature=0.1, top_p=0.9)
        response = outputs[0]["generated_text"]
        return self._process_response(response)

    def _process_response(self, response: str) -> Tuple[int, float]:
        """Elabora la risposta del modello per ottenere la predizione e la confidenza"""
        response = response.strip().upper()
        base_confidence = 0.8

        if "VISHING" in response:
            return 1, base_confidence
        elif "LEGITIMATE" in response:
            return 0, base_confidence
        else:
            return (1, base_confidence * 0.75) if "FRAUD" in response or "SCAM" in response else (0, base_confidence * 0.75)

    def classify_conversations(self, conversations: List[str], strategy: int, summarized: bool = False) -> Tuple[List[int], List[float]]:
        """Classifica una lista di conversazioni"""
        predictions = []
        confidences = []

        for text in tqdm(conversations, desc="Classificazione conversazioni", unit="conversazione"):
            pred, conf = self.classify_single(text, strategy, summarized)
            predictions.append(pred)
            confidences.append(conf)

        return predictions, confidences

def main(model_name: str):
    """
    Esegue la classificazione usando il modello Llama-3.2-1B-Instruct

    Args:
        model_name (str): Nome del modello da utilizzare
    """
    df = pd.read_csv(
        '../datasets/dataset_tradotto1.3.csv',
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
