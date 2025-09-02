### Min
import os
import pandas as pd
from rdkit import Chem

def is_valid_smiles(smiles):
### Check if the SMILES string is valid ===>
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def clean_dataset(input_csv_path, output_csv_path):
### Read the original data ===>
    df = pd.read_csv(input_csv_path)
    print(f"Original dataset size: {len(df)}")

### Check SMILES validity ===>
    df['valid_smiles'] = df['SMILES'].apply(is_valid_smiles)
    df_clean = df[df['valid_smiles']].drop(columns=['valid_smiles']).reset_index(drop=True)
    print(f"Number of valid SMILES: {len(df_clean)}")

### Save the cleaned data ===>
    df_clean.to_csv(output_csv_path, index=False)
    print(f"Cleaned data saved to: {output_csv_path}")

if __name__ == "__main__":
### Set up data paths ===>
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    input_path = os.path.join(data_dir, 'train.csv')
    output_path = os.path.join(data_dir, 'train_cleaned.csv')

    clean_dataset(input_path, output_path)

### Check missing values in the cleaned data ===>
    df = pd.read_csv(output_path)
    print("Missing values per column after cleaning:")
    print(df.isna().sum())
### #%#