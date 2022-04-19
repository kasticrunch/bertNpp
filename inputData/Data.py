def getData(csv):
    #testDf = pd.read_csv('testDataset.csv',usecols=['text','target'])
    #testDataset = Dataset.from_pandas(testDf)
    #testDataset = load_dataset('csv', data_files='testDataset.csv')
    
    df = pd.read_csv(csv,usecols=['text','target'])
    df = df.dropna()
    df = df.rename(columns={'target':'labels'})
    df['labels'] = df['labels'].apply(int)
    df.labels.value_counts().plot(kind='bar')
    df = df.sample(frac=1)
    return