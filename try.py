

def k_fold_generator(X, y, k_fold):
    subset_size = len(X) / k_fold  # Cast to int if using Python 3
    for k in range(k_fold):
        X_train = X[:k * subset_size] + X[(k + 1) * subset_size:]
        X_valid = X[k * subset_size:][:subset_size]
        y_train = y[:k * subset_size] + y[(k + 1) * subset_size:]
        y_valid = y[k * subset_size:][:subset_size]

        yield X_train, y_train, X_valid, y_valid


X = range(0,10)
Y = range(20,30)


Z = range(10,20)

for xt, yt, xv, yv in k_fold_generator(X, Y, 10):
	print(xt, yt,xv,yv)

for xt, yt, xv, yv in k_fold_generator(Z, Y, 10):
	print(xt, yt,xv,yv)