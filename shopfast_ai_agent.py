import pandas as pd

try:

    df = pd.read_csv("mock_intents.csv", delimiter=',', encoding='ISO-8859-1', header=None, skiprows=1)

    
    df[['User_Query', 'Intent']] = df[0].str.split(',', n=1, expand=True)
    df = df.drop(columns=[0])
    print("Data after final cleanup:")
    print(df.head())

except UnicodeDecodeError:
    print("UnicodeDecodeError: The file could not be read due to encoding issues.")
except FileNotFoundError:
    print("FileNotFoundError: The specified file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
