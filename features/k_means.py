import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
#X=2次元train_data,出力時に整形して1次元にする
class KMeansFeaturizer:
    """
    数値データをk-meansのクラスタIDに変換します。
    
    この変換器は入力データに対してk-meansを実行し、各データポイントを最も近い
    クラスタのIDに変換します。ターゲット変数yが存在する場合、クラス分類の境界を
    より重視したクラスタリングの結果を得るために、ターゲット変数をスケーリングして
    入力データに含めてk-meansに渡します。
    """
    #k=出力するクラスタの数

    def __init__(self, k=100, target_scale=5.0, random_state=None):
        self.k = k
        self.target_scale = target_scale
        self.random_state = random_state
        # np.array(range(k)).reshape(-1,1) ->2次元配列にそれぞれ一個ずつ要素が入っている。([[0],[1]...])
        #要素が入ってきたとき、one-hotvectorに変換する?([0]->[1,0,0...])
        self.cluster_encoder = OneHotEncoder(categories='auto').fit(
            np.array(range(k)).reshape(-1, 1))

    def fit(self, X, y=None):
        """
        入力データに対しk-meansを実行し、各クラスタの中心を見つけます。
        """
        if y is None:
            # ターゲット変数がない場合、ふつうのk-meansを実行します。
            km_model = KMeans(n_clusters=self.k, n_init=20,
                              random_state=self.random_state)
            km_model.fit(X)
            self.km_model = km_model
            self.cluster_centers_ = km_model.cluster_centers_
            return self

        # ターゲット変数がある場合。スケーリングして入力データに含めます。
        data_with_target = np.hstack((X, y[:, np.newaxis] * self.target_scale))

        # ターゲットを組み入れたデータで事前学習するためのk-meansモデルを構築します。
        km_model_pretrain = KMeans(
            n_clusters=self.k, n_init=20, random_state=self.random_state)
        km_model_pretrain.fit(data_with_target)

        # ターゲット変数の情報を除いて元の空間におけるクラスタを得るために
        # k-meansを再度実行します。事前学習で見つけたクラスタの中心を
        # 使って初期化し、クラスタの割り当てと中心の再計算を1回にします。
        km_model = KMeans(n_clusters=self.k, init=km_model_pretrain.cluster_centers_[
                          :, :2], n_init=1, max_iter=1)
        km_model.fit(X)

        self.km_model = km_model
        self.cluster_centers_ = km_model.cluster_centers_
        return self

    def transform(self, X, y=None):
        """
        入力データポイントに最も近いクラスタのIDを返します。
        """
        clusters = self.km_model.predict(X)
        #clustersが[[0],[8]...]みたいに代入されてそれがワンホットベクトルとして変換されて出力される?
        return self.cluster_encoder.transform(clusters.reshape(-1, 1))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    if __name__ == "__main__":
        import feather as ftr
        train = ftr.read_dataframe('../data/input/train.feather')
        
        train.drop(columns=['target', 'ID_code'], inplace=True)
        train = train.iloc[:10000, :]
        kmf = KMeansFeaturizer()
        k_train = kmf.fit_transform(train)
