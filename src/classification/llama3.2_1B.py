import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from typing import List, Tuple
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from tqdm import tqdm
import csv

# Modello specifico
MODEL_LLAMA_1B_INSTRUCT = "meta-llama/Llama-3.2-1B-Instruct"

class Llama1BInstructClassifier:
    """Classificatore specifico per il modello Llama-3.2-1B-Instruct"""

    def __init__(self, model_name: str, max_length: int = 2048):
        self.model_name = model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def truncate_text(self, text: str, prompt_template: str) -> str:
        """Tronca il testo per rispettare la lunghezza massima del modello"""
        prompt_tokens = len(self.tokenizer.encode(prompt_template))
        max_text_tokens = self.max_length - prompt_tokens - 50

        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_text_tokens:
            tokens = tokens[:max_text_tokens]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            text += " [TRUNCATED]"

        return text

    def summarize_conversation(self, text: str) -> str:
        """Genera una sintesi della conversazione"""
        prompt = f"""Summarize the following conversation in a single sentence:
        Conversation: {text}.
        Summary:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return summary

    def create_prompt(self, text: str, strategy: int) -> str:
        """Crea un prompt specifico in base alla strategia"""
        if strategy == 1:
            # Strategia 1: Solo contesto e conversazione
            template = """A vishing conversation is a type of phone fraud where a scammer pretends to be a trusted entity, like a bank or government agency, to trick the victim into providing sensitive information or performing specific actions.
            Analyze the following conversation and determine if it's a vishing attempt (phone fraud) or a legitimate conversation. Answer with 'VISHING' or 'LEGITIMATE'.
            Conversation: {conversation}.
            """
        elif strategy == 2:
            # Strategia 2: Esempio di conversazione vishing e conversazione target
            template = """Here is an example of a vishing conversation:
            Example: "A scammer, posing as Cheolmin Park from the Audit Department, repeatedly insists on resolving a "fraudulent transaction" involving the victim’s account. The caller uses complex and confusing language to create urgency, claiming issues with processing paperwork and requiring the victim to pay additional funds (e.g., $4 million won or $150) to resolve the matter, avoid penalties, and secure reimbursements. They repeatedly mention fraudulent payments, legal repercussions, and deadlines to pressure the victim into compliance, suggesting debit card reissues and delays to keep the victim engaged."
            Now analyze the following conversation and determine if it's a vishing attempt (phone fraud) or a legitimate conversation. Answer with 'VISHING' or 'LEGITIMATE'.
            Conversation: {conversation}.
            """
        else:
            raise ValueError("Strategia non valida. Usare 1 o 2.")

        return template.format(conversation=self.truncate_text(text, template))

    def classify_single(self, text: str, strategy: int, summarized: bool = False) -> Tuple[int, float]:
        """Classifica una singola conversazione"""
        if summarized:
            text = self.summarize_conversation(text)

        prompt = self.create_prompt(text, strategy)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
