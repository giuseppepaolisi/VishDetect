import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from typing import List, Tuple
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from abc import ABC, abstractmethod
from tqdm import tqdm

# Definizione dei modelli supportati
MODEL_TINY_LLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_GPT_NEO = "EleutherAI/gpt-neo-1.3B"
MODEL_PHI_2 = "microsoft/phi-2"

# Lista di tutti i modelli supportati
SUPPORTED_MODELS = [
    MODEL_TINY_LLAMA,
    MODEL_GPT_NEO,
    MODEL_PHI_2
]

class BaseVishingClassifier(ABC):
    """Base class per i classificatori di vishing"""
    
    def __init__(self, model_name: str, max_length: int = 2048):
        self.model_name = model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        
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
    
    @abstractmethod
    def create_prompt(self, text: str) -> str:
        """Crea un prompt specifico per il modello"""
        pass
    
    def classify_single(self, text: str) -> Tuple[int, float]:
        """Classifica una singola conversazione"""
        prompt = self.create_prompt(text)
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
    
    @abstractmethod
    def _process_response(self, response: str) -> Tuple[int, float]:
        """Elabora la risposta del modello per ottenere la predizione e la confidenza"""
        pass
    
    def classify_conversations(self, conversations: List[str]) -> Tuple[List[int], List[float]]:
        """Classifica una lista di conversazioni"""
        predictions = []
        confidences = []
        
        for text in tqdm(conversations, desc="Classificazione conversazioni", unit="conversazione"):
            pred, conf = self.classify_single(text)
            predictions.append(pred)
            confidences.append(conf)
            
        return predictions, confidences

class TinyLlamaClassifier(BaseVishingClassifier):
    """Implementazione specifica per TinyLlama"""
    
    def create_prompt(self, text: str) -> str:
        template = """<human>Analyze the following conversation and determine if it's a vishing attempt (phone fraud) or a legitimate conversation. 
        Please answer only with 'VISHING' if it's a phone fraud or 'LEGITIMATE' if it's a normal conversation.
        
        Conversation: </human>

        <assistant>Based on my analysis of the conversation, my answer is: """
        
        text = self.truncate_text(text, template)
        return f"""<human>Analyze the following conversation and determine if it's a vishing attempt (phone fraud) or a legitimate conversation. 
        Please answer only with 'VISHING' if it's a phone fraud or 'LEGITIMATE' if it's a normal conversation.
        
        Conversation: {text}</human>

        <assistant>Based on my analysis of the conversation, my answer is: """
    
    def _process_response(self, response: str) -> Tuple[int, float]:
        response = response.split("my answer is: ")[-1].strip().upper()
        base_confidence = 0.8
        
        if "VISHING" in response:
            return 1, base_confidence
        elif "LEGITIMATE" in response:
            return 0, base_confidence
        else:
            return (1, base_confidence * 0.75) if "FRAUD" in response or "SCAM" in response else (0, base_confidence * 0.75)

class GPTNeoClassifier(BaseVishingClassifier):
    """Implementazione specifica per GPT-Neo"""
    
    def create_prompt(self, text: str) -> str:
        template = """Analyze this conversation and determine if it's a vishing (phone fraud) attempt.
        Reply only with VISHING for phone fraud or LEGITIMATE for normal conversation.
        
        Conversation: """
        
        text = self.truncate_text(text, template)
        return f"""Analyze this conversation and determine if it's a vishing (phone fraud) attempt.
        Reply only with VISHING for phone fraud or LEGITIMATE for normal conversation.
        
        Conversation: {text}
        
        Analysis: """
    
    def _process_response(self, response: str) -> Tuple[int, float]:
        response = response.split("Analysis: ")[-1].strip().upper()
        base_confidence = 0.75
        
        if "VISHING" in response:
            return 1, base_confidence
        elif "LEGITIMATE" in response:
            return 0, base_confidence
        else:
            return (1, base_confidence * 0.75) if "FRAUD" in response or "SCAM" in response else (0, base_confidence * 0.75)

class Phi2Classifier(BaseVishingClassifier):
    """Implementazione specifica per Phi-2"""
    
    def create_prompt(self, text: str) -> str:
        template = """Instruct: Analyze the following conversation and classify it as either a vishing attempt (phone fraud) or legitimate conversation.
        Output only VISHING for phone fraud or LEGITIMATE for normal conversation.
        
        Conversation: 
        
        Output: """
        
        text = self.truncate_text(text, template)
        return f"""Instruct: Analyze the following conversation and classify it as either a vishing attempt (phone fraud) or legitimate conversation.
        Output only VISHING for phone fraud or LEGITIMATE for normal conversation.
        
        Conversation: {text}
        
        Output: """
    
    def _process_response(self, response: str) -> Tuple[int, float]:
        response = response.split("Output: ")[-1].strip().upper()
        base_confidence = 0.75
        
        if "VISHING" in response:
            return 1, base_confidence
        elif "LEGITIMATE" in response:
            return 0, base_confidence
        else:
            return (1, base_confidence * 0.75) if "FRAUD" in response or "SCAM" in response else (0, base_confidence * 0.75)

def get_classifier(model_name: str) -> BaseVishingClassifier:
    """Funzione factory per ottenere il classificatore appropriato in base al nome del modello"""
    classifiers = {
        MODEL_TINY_LLAMA: TinyLlamaClassifier,
        MODEL_GPT_NEO: GPTNeoClassifier,
        MODEL_PHI_2: Phi2Classifier
    }
    
    if model_name not in classifiers:
        raise ValueError(f"Model {model_name} non supportato. Modelli disponibili: {SUPPORTED_MODELS}")
        
    return classifiers[model_name](model_name)

def main(model_name: str):
    """
    Esegue la classificazione usando un singolo modello specificato
    
    Args:
        model_name (str): Nome del modello da utilizzare
    """
    df = pd.read_csv(
        '../datasets/EngCCViD_v1.3.csv',
        quoting=pd.io.common.QUOTE_NONNUMERIC,
        escapechar='\\',
        na_filter=False 
    )
    
    # Verifica che tutti i transcript siano stringhe
    df['Transcript'] = df['Transcript'].astype(str)
    
    # Inizializza il classificatore per il modello specifico
    classifier = get_classifier(model_name)
    
    print(f"\nUtilizzo del modello: {model_name}")
    
    # Effettua le predizioni
    predictions, confidences = classifier.classify_conversations(df['Transcript'].tolist())
    
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
    
    # Crea il nome del file di output con timestamp e nome del modello
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = model_name.split('/')[-1]
    output_file = f'predictions_{model_short_name}_{timestamp}.csv'
    
    # Salva il DataFrame in CSV
    results_df.to_csv(
        output_file,
        index=False,
        quoting=pd.io.common.QUOTE_NONNUMERIC,  # Forza il quoting per campi non numerici
        escapechar='\\',  # Gestisce eventuali caratteri di escape
        na_rep=''        # Gestisce i valori NA/NaN
    )
    
    print(f"\nRisultati salvati in: {output_file}")
    
    return results_df

if __name__ == "__main__":
    results_df = main(MODEL_GPT_NEO)