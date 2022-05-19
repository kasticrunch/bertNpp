import pandas as pd
def getData(csv):
    df = pd.read_csv(csv,usecols=['text','target'])
    df = df.dropna()
    df = df.rename(columns={'target':'labels'})
    df['labels'] = df['labels'].apply(int)
    df.labels.value_counts().plot(kind='bar')
    df = df.sample(frac=1)
    return
