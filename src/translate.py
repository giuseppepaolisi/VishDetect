import pandas as pd
import deepl
import time

import os
from dotenv import load_dotenv

load_dotenv()
auth_key = os.getenv('TOKEN_DEEPL')
translator = deepl.Translator(auth_key)

df = pd.read_csv('KorCCViD_v1.3.csv')

# Funzione per tradurre il testo
def traduci_testo(testo):
    if pd.isna(testo):
        return testo
    try:
        result = translator.translate_text(testo, target_lang='EN-US')
        return result.text
    except deepl.DeepLException as e:
        print(f"Errore nella traduzione del testo: {testo}\nErrore: {e}")
        return testo

# Numero di caratteri tradotti (limite versione gratuita 500000)
caratteri_tradotti = 0

for index, row in df.iterrows():
    testo = row['Transcript']
    if pd.isna(testo):
        continue

    # Verifica se la traduzione supererebbe il limite mensile
    if caratteri_tradotti + len(testo) > 500000:
        print("Limite di caratteri raggiunto. Interruzione del processo.")
        break

    traduzione = traduci_testo(testo)
    df.at[index, 'Transcript'] = traduzione

    caratteri_tradotti += len(testo)

    # Salva i progressi ogni 100 traduzioni
    if index % 100 == 0:
        df.to_csv('dataset_tradotto.csv', index=False)
        print(f"Salvati i progressi fino alla riga {index}.")

    # Aggiungi un ritardo tra le richieste per evitare di superare i limiti dell'API
    time.sleep(1)

# Salva il dataset tradotto
df.to_csv('dataset_tradotto.csv', index=False)
print("Traduzione completata e file salvato.")