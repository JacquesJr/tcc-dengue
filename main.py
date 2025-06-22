from src import preprocess_data, predict_deng, load_and_merge_data

def main():
    df_processed = load_and_merge_data()
    print(df_processed.shape)
    predict_deng()

if __name__ == "__main__":
    main()
