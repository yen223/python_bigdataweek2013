from sklearn import decomposition
import datas


def main():
    training = datas.training
    pca = decomposition.PCA()
    pca.fit(training.ix[:,'pclass':])
    result =  pca.explained_variance_
    print result
    pca.n_components = 3
    reduced = pca.fit_transform(training.ix[:,'pclass':])
    print reduced.shape


if __name__ == "__main__":
    main()
