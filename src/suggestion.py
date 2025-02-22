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
                Context:
                You are provided with a transcript from a vishing (voice phishing) attempt. Vishing is a deceptive practice where fraudsters use phone calls to trick individuals into disclosing sensitive personal or financial information. Your task is to thoroughly analyze the provided conversation transcript and generate a set of targeted, actionable suggestions designed to help potential victims defend themselves against this specific vishing attempt.

                Instructions:

                    Analyze the Conversation:
                        Examine for Red Flags: Carefully review the transcript for any suspicious elements such as urgent language, pressure tactics, or requests for sensitive data.
                        Identify Key Moments: Pinpoint specific segments where the attacker employs strategies to deceive or pressure the victim.

                    Provide Targeted, Actionable Suggestions:
                        List Clear Recommendations: Offer detailed suggestions that directly address the red flags and tactics identified in the conversation.
                        Include Action and Relevance for Each Suggestion:
                            Action: Specify the exact step a potential victim should take.
                            Relevance: Explain why this step is important in countering the tactics used by the scammer.

                Example:

                Sample Conversation:
                "They have illegally detained it inside. We did it, so if we send this regarding that point, will there be any issues with the pelvis or anything unusual? It will work. For now, our prosecution will proceed with the Financial Supervisory Service and focus on the accounts. The content can be separated for the Financial Supervisory Service certification. Please use Nonghyup once. I have never used Hana Bank. Just do it yourself and briefly provide the name of the museum where you are using it in the financial sector. Please prepare the business name for Falun Gong like this for our daughter-in-law."

                Example Suggestions:

                Verify Caller Identity Independently
                Action: If you receive a call using official-sounding terms (e.g., referencing the Financial Supervisory Service), immediately end the conversation and contact the institution using verified, official phone numbers or websites.
                Relevance: Attackers often impersonate authorities to gain trust. Independently verifying the caller’s identity prevents you from falling prey to false authority.

                Do Not Provide Personal or Unusual Business Information
                Action: Refuse to share any personal details, business names, or specific operational information, especially when the requests seem unrelated or nonsensical.
                Relevance: Requests such as “provide the name of the museum” or “prepare the business name for Falun Gong” are red flags. Legitimate institutions do not require such arbitrary information over unsolicited calls.

                Recognize and Question Incoherent or Pressure Tactics
                Action: Be alert when language or instructions seem confusing or forced. Politely ask for clarification or, if the response remains unclear, terminate the call.
                Relevance: Confusing language (e.g., “issues with the pelvis”) is likely used to overwhelm you, making it easier for the scammer to exploit your uncertainty.

                Avoid Following Unverified Banking Instructions
                Action: Do not follow instructions to “use” a particular bank or to avoid others (such as the comment regarding Hana Bank). Instead, consult your bank directly using their official communication channels.
                Relevance: Scammers may try to divert your transactions or manipulate your banking behavior. Always use your own verified channels for financial decisions.

                Do Not Succumb to Urgency or Threats
                Action: Remain calm if the caller invokes legal action or regulatory involvement. Take a moment to verify the claims independently rather than reacting immediately.
                Relevance: Urgency and threats are classic pressure tactics used in vishing to compel quick, unthoughtful responses that could compromise your personal or financial security.

                Report Suspicious Activity
                Action: Document the details of the call and report it to your financial institution and relevant local authorities or consumer protection agencies.
                Relevance: Reporting can help prevent others from falling victim and may assist authorities in tracking fraudulent activities.

                Educate Yourself About Vishing Tactics
                Action: Regularly update your knowledge about common vishing scams by consulting trusted resources such as official government or consumer protection websites.
                Relevance: Awareness of vishing techniques can empower you to recognize red flags and protect your sensitive information in the future.

                Task:
                Below is a vishing conversation transcript that requires your analysis. Carefully review the dialogue, highlight the suspicious elements, and provide detailed, conversation-specific suggestions. For every recommendation, include both the Action (what to do) and the Relevance (why it is important) to ensure clarity and practical guidance for potential victims.
                """
            )
            prompt = (
                f"{system_content}\n\n"
                f"Analyze this conversation:\n«{text}»\n\n"
                "Your response should include a thorough analysis of the transcript with specific, actionable recommendations that directly address the tactics used in the conversation. Each suggestion must include both the Action and the Relevance for a clear understanding of why the advice is essential.\n"
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
            max_new_tokens=200,
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

        for text in tqdm(conversations, desc="conversations", unit="conversation"):
            text, suggestion = self.classify_single(text)
            texts.append(text)
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
    dataset_path = os.path.join(script_dir, '../datasets/new_dataset.csv')
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
    output_file = f'results_{model_short_name}_{timestamp}.csv'

    # Salva il DataFrame in CSV
    results_df.to_csv(
        output_file,
        index=False,
        sep=',',
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        escapechar='\\',
        line_terminator='\n',
        na_rep=''
    )

    print(f"\nRisultati salvati in: {output_file}")

if __name__ == "__main__":
    main(MODEL_PHI_3)