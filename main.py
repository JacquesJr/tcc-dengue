from src import preprocess_data, predict_deng

def main():
    df_merged = preprocess_data()
    print(df_merged.shape)
    predict_deng()

if __name__ == "__main__":
    main()
