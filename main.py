from src import load_data, preprocess_and_merge, train_random_forest, prepocess_uberaba_data, prever_casos

def main():
    # df_dengue, df_cities = load_data()
    # df_merged = preprocess_and_merge(df_dengue, df_cities)

    # model, mse, importancias = train_random_forest(df_merged)
    # print(model)
    # print(mse)
    # print(importancias)

    df_merged = prepocess_uberaba_data()
    model, y_test, y_pred, mae = prever_casos(df_merged)
    print('model: ', model)
    print('y_test: ', y_test)
    print('y_pred: ', y_pred)
    print('mae: ', mae)

if __name__ == "__main__":
    main()
