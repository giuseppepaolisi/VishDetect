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

    def create_message(self, text: str) -> str:
            """Format prompt according to DeepSeek-R1 specifications"""
            system_content = (
                """
                **Context:**  
                You are given a transcript of a vishing (voice phishing) attempt. Vishing is a deceptive practice where fraudsters use phone calls to manipulate individuals into disclosing sensitive personal or financial information. Your task is to thoroughly analyze the transcript and generate a set of targeted, actionable suggestions to help potential victims protect themselves from this specific vishing attempt.  

                **Task:**  
                Analyze the vishing call transcript and create **five specific defense tips** for non-experts. Ensure all locations, names, and companies are anonymized.  

                **Requirements:**  
                1. Identify and list the sentences in the transcript that indicate vishing.  
                2. Explain why the conversation qualifies as vishing.  
                3. Generate **exactly five** protective measures tailored to the transcript.  
                4. Format each suggestion as follows:  

                  **[Number]. [Short Title]**  
                  **Action:** [Clear, executable instruction]  

                **Analysis Framework:**  
                a) Use of pressure tactics (urgency, threats)  
                b) Suspicious requests (personal/financial information)  
                c) Authority impersonation  
                d) Unusual payment or banking instructions  
                e) Incoherent or misleading narratives  

                **Strict Rules:**  
                - **Provide exactly five** numbered suggestions.  
                - **Do not use markdown, headers, or bullet points** in the response.  
                - Use anonymous references (e.g., "the institution" instead of specific names).  
                - Ensure that all action steps are clear and executable for non-technical users.  

                **Example:**  

                *Conversation:*  
                "They have illegally detained it inside. We did it, so if we send this regarding that point, will there be any issues with the pelvis or anything unusual? It will work. For now, our prosecution will proceed with the Financial Supervisory Service and focus on the accounts. The content can be separated for the Financial Supervisory Service certification. Please use Nonghyup once. I have never used Hana Bank. Just do it yourself and briefly provide the name of the museum where you are using it in the financial sector. Please prepare the business name for Falun Gong like this for our daughter-in-law."  
                Exemple result:
                *Analysis:*  

                *Sentences indicating vishing*:
                "They have illegally detained it inside. We did it, so if we send this regarding that point, will there be any issues with the pelvis or anything unusual? It will work."
                "For now, our prosecution will proceed with the supervisory service and focus on the accounts."
                "The content can be separated for the supervisory service certification."
                "Please use a bank once. I have never used another bank."
                "Just do it yourself and briefly provide the name of the museum where you are using it in the financial sector. Please prepare the business name for an organization like this for our family member."

                This conversation qualifies as vishing because it impersonates an authority, uses confusing language and pressure tactics, and makes suspicious requests for financial and personal information.

                Verify Caller Identity
                Action: Contact the institution directly using official contact details to confirm the caller’s identity.

                Do Not Share Sensitive Information
                Action: Refuse to provide personal, financial, or business details to unsolicited callers.

                Request Written Confirmation
                Action: Ask for clear, written instructions or official documentation before taking any action.

                Report Suspicious Communications
                Action: Immediately report any dubious calls to local law enforcement or a trusted fraud prevention agency.

                Seek Independent Advice
                Action: Consult a trusted advisor or legal expert to verify the legitimacy of any requests made during the call.                
                """
            )

            prompt = (
                f"{system_content}\n\n"
                f"Conversation: «{text}»\n\n"
                "Analyze the conversation and list sentences that indicate vishing. "
                "Explain why this conversation qualifies as vishing. "
                "Provide exactly five numbered suggestions. Each must include:\n"
                "- A short title\n"
                "- A concrete action step\n"
                "Use anonymous references. No explanations beyond the required fields."
            )


            return prompt

    def classify_single(self, text: str) -> Tuple[str, str]:
        """Classify a single conversation."""
        prompt = self.create_message(text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.6,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return text, response

    def classify_conversations(self, conversations: List[str]) -> pd.DataFrame:
        """Classify a list of conversations."""
        texts = []
        suggestions = []
        conversations = conversations[:10]
        print(conversations)

        for text in tqdm(conversations, desc="conversations", unit="conversation"):
            text, suggestion = self.classify_single(text)
            texts.append(text)
            suggestion = ''.join(c for c in suggestion if c not in '\n"#$*')
            suggestions.append(suggestion)

        return pd.DataFrame({
            'text': texts,
            'suggestion': suggestions
        })

def main(model_name: str):
    """
    Esegue la classificazione usando il modello Llama-3.2-1B-Instruct.

    Args:
        model_name (str): Nome del modello da utilizzare.
    """
    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join('../datasets/new_dataset.csv')
    print(dataset_path)
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

    # Esegui la classificazione
    results_df = classifier.classify_conversations(df['Transcript'].tolist())

    # Stampa i risultati
    print("\nRisultati:")
    print(results_df)

    # Crea il nome del file di output con timestamp, strategia e modalità
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = model_name.split('/')[-1]
    output_file = f'results_{model_short_name}.csv'

    # Salva il DataFrame in CSV
    results_df.to_csv(
        output_file,
        index=False,
        sep=',',
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        escapechar='\\',
        lineterminator='\n',
        na_rep=''
    )

    print(f"\nRisultati salvati in: {output_file}")

if __name__ == "__main__":
    main(MODEL_PHI_3)