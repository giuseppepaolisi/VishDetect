import pandas as pd
import numpy as np
import os

def main():
    # Carica i dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        df1 = pd.read_csv(os.path.join(script_dir, '../../datasets/dataset1.3.csv'), encoding='latin1')
    except Exception as e:
        print(f"Error loading dataset1.3.csv: {e}")
        return

    try:
        df2 = pd.read_csv(os.path.join(script_dir, '../../datasets/dataset2.1.csv'), encoding='latin1')
    except Exception as e:
        print(f"Error loading dataset2.1.csv: {e}")
        return

    # Elimina la colonna "id" dal dataset2.1
    df2 = df2.iloc[:, 1:]

    # Estrai le istanze con Label = 1
    df1_label1 = df1[df1['Label'] == 1]
    df2_label1 = df2[df2['Label'] == 1]

    # Estrai 30 istanze casuali da ogni dataset
    df1_samples = df1_label1.sample(n=30, random_state=42)
    df2_samples = df2_label1.sample(n=30, random_state=42)

    # Crea il nuovo dataset
    new_df = pd.concat([df1_samples, df2_samples], ignore_index=True)

    # Salva il nuovo dataset
    new_df.to_csv(os.path.join(script_dir, '../../datasets/new_dataset.csv'), index=False)

if __name__ == '__main__':
    main()
