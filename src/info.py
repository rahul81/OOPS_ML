class Information:
    def __init__(self):

        print("Information")

    def _get_missing_values(self, data):

        # Getting sum of missing values for each feature
        missing_values = data.isnull().sum()
        # Feature missing values are sorted from few to many
        missing_values.sort_values(ascending=False, inplace=True)

        # Returning missing values
        return missing_values

    def info(self, data):

        feature_dtypes = data.dtypes
        self.missing_values = self._get_missing_values(data)

        print("=" * 50)

        print(
            "{:16} {:16} {:25} {:16}".format(
                "Feature Name".upper(),
                "Data Format".upper(),
                "# of Missing Values".upper(),
                "Samples".upper(),
            )
        )
        for feature_name, dtype, missing_value in zip(
            self.missing_values.index.values,
            feature_dtypes[self.missing_values.index.values],
            self.missing_values.values,
        ):
            print(
                "{:18} {:19} {:19} ".format(
                    feature_name, str(dtype), str(missing_value)
                ),
                end="",
            )
            for v in data[feature_name].values[:10]:
                print(v, end=",")
            print()

        print("=" * 50)

        print("Training data :")
        print(data.head())
