import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from typing import List, Tuple
from sklearn.metrics import classification_report, confusion_matrix

def parse_dataset_line(line: str) -> Tuple[str, int]:
    """
    Analizza una linea del dataset nel formato specifico
    
    Args:
        line (str): Linea del dataset
        
    Returns:
        Tuple[str, int]: (testo, label)
    """
    parts = line.strip().split('","')
    if len(parts) == 2:
        text = parts[0].lstrip('"')
        label = int(parts[1].rstrip('"'))
        return text, label
    else:
        raise ValueError(f"Formato linea non valido: {line}")

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Carica il dataset dal file
    
    Args:
        file_path (str): Percorso del file
        
    Returns:
        pd.DataFrame: DataFrame con le colonne Transcript e Label
    """
    texts = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        pattern = r'"([^"]+)","(\d+)"'
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            text, label = match
            texts.append(text)
            labels.append(int(label))
    
    return pd.DataFrame({
        'Transcript': texts,
        'Label': labels
    })

class TinyLlamaVishingClassifier:
    def __init__(self):
        """
        Inizializza il classificatore usando TinyLlama
        """
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.max_length = 2048
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
    
    def truncate_text(self, text: str) -> str:
        """
        Tronca il testo per rispettare la lunghezza massima del modello
        """
        prompt_template = """<human>Analyze the following conversation and determine if it's a vishing attempt (phone fraud) or a legitimate conversation. 
        Please answer only with 'VISHING' if it's a phone fraud or 'LEGITIMATE' if it's a normal conversation.
        
        Conversation: </human>

        <assistant>Based on my analysis of the conversation, my answer is: """
        
        prompt_tokens = len(self.tokenizer.encode(prompt_template))
        max_text_tokens = self.max_length - prompt_tokens - 50
        
        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_text_tokens:
            tokens = tokens[:max_text_tokens]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            text += " [TRUNCATED]"
        
        return text
        
    def create_prompt(self, text: str) -> str:
        """
        Crea il prompt per la classificazione zero-shot

        Args:
            text (str): Testo da classificare
        """
        text = self.truncate_text(text)
        
        return f"""<human>Analyze the following conversation and determine if it's a vishing attempt (phone fraud) or a legitimate conversation. 
        Please answer only with 'VISHING' if it's a phone fraud or 'LEGITIMATE' if it's a normal conversation.
        
        Conversation: {text}</human>

        <assistant>Based on my analysis of the conversation, my answer is: """
    
    def classify_single(self, text: str) -> Tuple[int, float]:
        """
        Classifica una singola conversazione
        """
        prompt = self.create_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("my answer is: ")[-1].strip().upper()
        
        base_confidence = 0.8
        
        if "VISHING" in response:
            return 1, base_confidence
        elif "LEGITIMATE" in response:
            return 0, base_confidence
        else:
            return (1, base_confidence * 0.75) if "FRAUD" in response or "SCAM" in response else (0, base_confidence * 0.75)

    def classify_conversations(self, conversations: List[str]) -> Tuple[List[int], List[float]]:
        """
        Classifica una lista di conversazioni
        """
        predictions = []
        confidences = []
        
        for text in conversations:
            pred, conf = self.classify_single(text)
            predictions.append(pred)
            confidences.append(conf)
            
        return predictions, confidences

def main():
    # Carica il dataset
    df = pd.read_csv('../datasets/EngCCViD_v1.3.csv')
    
    # Inizializza il classificatore
    classifier = TinyLlamaVishingClassifier()
    
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
    print("\nClassification Report:")
    print(classification_report(df['Label'], predictions))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(df['Label'], predictions))
    
    # Crea il nome del file di output con timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'predictions_{timestamp}.csv'
    
    # Salva il DataFrame in CSV
    results_df.to_csv(output_file, index=False)
    print(f"\nRisultati salvati in: {output_file}")
    
    return results_df

if __name__ == "__main__":
    results_df = main()
