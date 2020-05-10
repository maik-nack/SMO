import numpy as np


# k(x1, x2) = <x1, x2>
def linear_kernel(x1, x2):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    return x1.dot(x2.T)


# k(x, y) = exp(- gamma ||x1 - x2||^2)
def get_rbf_kernel(gamma):
    def rbf_kernel(x1, x2):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        s1, _ = x1.shape
        s2, _ = x2.shape
        norm1 = np.ones((s2, 1)).dot(np.atleast_2d(np.sum(x1 ** 2, axis=1))).T
        norm2 = np.ones((s1, 1)).dot(np.atleast_2d(np.sum(x2 ** 2, axis=1)))
        return np.exp(- gamma * (norm1 + norm2 - 2 * x1.dot(x2.T)))
    return rbf_kernel


# k(x1, x2) = (<x1, x2> + coef0)^degree
def get_polynomial_kernel(degree, coef0):
    def polynomial_kernel(x1, x2):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        return (x1.dot(x2.T) + coef0) ** degree
    return polynomial_kernel


class SVM:

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=1e-3, max_iter=-1):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.__eps = 1e-7
        self.__kernel_func = None
        self.__alpha = None
        self.__fcache = None
        self.__i_low, self.__i_up = None, None
        self.__b_low, self.__b_up = None, None
        self.__I_0, self.__I_1, self.__I_2, self.__I_3, self.__I_4 = None, None, None, None, None
        self.__coef = None
        self.__dual_coef = None
        self.__threshold = None
        self.support = None
        self.support_vectors = None

    def __get_gamma(self, X):
        if isinstance(self.gamma, float):
            return self.gamma
        elif self.gamma == 'auto':
            return 1.0 / X.shape[1]
        elif self.gamma == 'scale':
            X_var = X.var()
            return 1.0 / (X.shape[1] * X_var) if X_var > self.__eps else 1.0
        else:
            raise ValueError(f"'{self.gamma}' is incorrect value for gamma")

    def __get_kernel_function(self, X):
        if callable(self.kernel):
            return self.kernel
        elif self.kernel == 'linear':
            return linear_kernel
        elif self.kernel == 'rbf':
            return get_rbf_kernel(self.__get_gamma(X))
        elif self.kernel == 'poly':
            return get_polynomial_kernel(self.degree, self.coef0)
        else:
            raise ValueError(f"'{self.kernel}' is incorrect value for kernel")

    def __compute_L_H(self, y1, y2, alpha1, alpha2):
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)
        return L, H

    def __compute_objective_function(self, y1, y2, F1, F2, alpha1, alpha2, s, k11, k12, k22, L, H):
        f1 = y1 * F1 - alpha1 * k11 - s * alpha2 * k12
        f2 = y2 * F2 - s * alpha1 * k12 - alpha2 * k22
        L1 = alpha1 + s * (alpha2 - L)
        H1 = alpha1 + s * (alpha2 - H)
        Psi_L = L1 * f1 + L * f2 + 0.5 * L1 ** 2 * k11 + 0.5 * L ** 2 * k22 + s * L * L1 * k12
        Psi_H = H1 * f1 + H * f2 + 0.5 * H1 ** 2 * k11 + 0.5 * H ** 2 * k22 + s * H * H1 * k12
        return Psi_L, Psi_H

    def __update_I(self, i, y, a):
        if self.__I_0[i]:
            self.__I_0[i] = False
        else:
            if y == 1:
                if self.__I_1[i]:
                    self.__I_1[i] = False
                else:
                    self.__I_3[i] = False
            else:
                if self.__I_2[i]:
                    self.__I_2[i] = False
                else:
                    self.__I_4[i] = False
        if a <= self.__eps or a >= self.C - self.__eps:
            if y == 1:
                if a <= self.__eps:
                    self.__I_1[i] = True
                else:
                    self.__I_3[i] = True
            else:
                if a <= self.__eps:
                    self.__I_4[i] = True
                else:
                    self.__I_2[i] = True
        else:
            self.__I_0[i] = True

    def __update_I_low_up(self, I_low, I_up, i):
        if self.__I_3[i] or self.__I_4[i]:
            I_low[i] = True
        else:
            I_up[i] = True

    def __get_b_i(self, I, argfunc):
        I = np.where(I)[0]
        F = self.__fcache[I]
        i = I[argfunc(F)]
        b = self.__fcache[i]
        return b, i

    def __take_step(self, i1, i2, X, y):
        if i1 == i2:
            return False
        y1 = y[i1]
        y2 = y[i2]
        alpha1 = self.__alpha[i1]
        alpha2 = self.__alpha[i2]
        F1 = self.__fcache[i1]
        F2 = self.__fcache[i2]
        s = y1 * y2
        L, H = self.__compute_L_H(y1, y2, alpha1, alpha2)  # Compute L and H
        if abs(L - H) < self.__eps:
            return False
        k11 = self.__kernel_func(X[i1], X[i1])[0, 0]
        k12 = self.__kernel_func(X[i1], X[i2])[0, 0]
        k22 = self.__kernel_func(X[i2], X[i2])[0, 0]
        eta = 2 * k12 - k11 - k22
        if eta < -self.__eps:
            a2 = alpha2 - y2 * (F1 - F2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            # Compute objective function at a2 = L and a2 = H
            L_obj, H_obj = self.__compute_objective_function(y1, y2, F1, F2, alpha1, alpha2, s, k11, k12, k22, L, H)
            if L_obj < H_obj - self.__eps:
                a2 = L
            elif L_obj > H_obj + self.__eps:
                a2 = H
            else:
                a2 = alpha2
        if abs(a2 - alpha2) < self.__eps * (a2 + alpha2 + self.__eps):
            return False
        a1 = alpha1 + s * (alpha2 - a2)
        # Update fcache[i] for i in I_0 using new Lagrange multipliers
        ki1 = self.__kernel_func(X[self.__I_0], X[i1]).ravel()
        ki2 = self.__kernel_func(X[self.__I_0], X[i2]).ravel()
        self.__fcache[self.__I_0] += y1 * (a1 - alpha1) * ki1 + y2 * (a2 - alpha2) * ki2
        # Store a1 and a2 in the alpha array
        self.__alpha[i1] = a1
        self.__alpha[i2] = a2
        # Update I_0, I_1, I_2, I_3 and I_4
        self.__update_I(i1, y1, a1)
        self.__update_I(i2, y2, a2)
        # Compute updated F values for i1 and i2
        self.__fcache[i1] = F1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12
        self.__fcache[i2] = F2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22
        # Compute (i_low, b_low) and (i_up, b_up)
        I_low, I_up = self.__I_0.copy(), self.__I_0.copy()
        self.__update_I_low_up(I_low, I_up, i1)
        self.__update_I_low_up(I_low, I_up, i2)
        self.__b_low, self.__i_low = self.__get_b_i(I_low, np.argmax)
        self.__b_up, self.__i_up = self.__get_b_i(I_up, np.argmin)
        return True

    def __compute_F(self, X, y, i):
        return np.sum(self.__kernel_func(X[i], X).ravel() * (self.__alpha * y)) - y[i]

    def __examine_example(self, i2, X, y):
        if self.__I_0[i2]:
            F2 = self.__fcache[i2]
        else:
            F2 = self.__compute_F(X, y, i2)  # compute F_i2
            self.__fcache[i2] = F2
            # Update (b_low, i_low) or (b_up, i_up) using (F2, i2)
            if (self.__I_1[i2] or self.__I_2[i2]) and F2 < self.__b_up:
                self.__b_up, self.__i_up = F2, i2
            elif (self.__I_3[i2] or self.__I_4[i2]) and F2 > self.__b_low:
                self.__b_low, self.__i_low = F2, i2
        # Check optimality using current b_low and b_up
        # If violated, find an index i1 to do joint optimization with i2
        optimality = True
        i1 = 0
        if (self.__I_0[i2] or self.__I_1[i2] or self.__I_2[i2]) and self.__b_low - F2 > 2 * self.tol:
            optimality = False
            i1 = self.__i_low
        if (self.__I_0[i2] or self.__I_3[i2] or self.__I_4[i2]) and F2 - self.__b_up > 2 * self.tol:
            optimality = False
            i1 = self.__i_up
        if optimality:
            return 0
        # For i2 in I_0 choose the better i1
        if self.__I_0[i2]:
            if self.__b_low - F2 > F2 - self.__b_up:
                i1 = self.__i_low
            else:
                i1 = self.__i_up
        return int(self.__take_step(i1, i2, X, y))

    def __initialize_fitting(self, X, y):
        X, y = np.asarray(X), np.asarray(y)

        self.__alpha = np.zeros(y.shape[0])
        self.__fcache = np.zeros(y.shape[0])

        self.__b_up = -1
        y1 = y == 1
        self.__i_up = y1.nonzero()[0][0]
        self.__I_1 = y1

        self.__b_low = 1
        y2 = y == -1
        self.__i_low = y2.nonzero()[0][0]
        self.__I_4 = y2

        self.__fcache[self.__i_low] = 1
        self.__fcache[self.__i_up] = -1

        self.__I_0, self.__I_2, self.__I_3 = np.zeros(y.shape, bool), np.zeros(y.shape, bool), np.zeros(y.shape, bool)

        self.__kernel_func = self.__get_kernel_function(X)
        max_iter = self.max_iter if self.max_iter >= 0 else np.inf

        return X, y, max_iter

    def __set_result(self, X, y):
        self.support = np.where(self.__alpha > self.__eps)[0]
        self.support_vectors = X[self.support]
        self.__dual_coef = self.__alpha[self.support] * y[self.support]
        self.__threshold = (self.__b_low + self.__b_up) / 2
        if self.kernel == 'linear':
            self.__coef = np.atleast_2d(np.sum(self.__dual_coef * self.support_vectors.T, axis=1))

    def fit_modification1(self, X, y):
        X, y, max_iter = self.__initialize_fitting(X, y)
        iteration = 0
        num_changed = 0
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(y.shape[0]):
                    num_changed += self.__examine_example(i, X, y)
            else:
                for i in np.where(self.__I_0)[0]:
                    num_changed += self.__examine_example(i, X, y)
                    if self.__b_up > self.__b_low - 2 * self.tol:
                        num_changed = 0
                        break
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            iteration += 1
            if iteration >= max_iter:
                break
        self.__set_result(X, y)

    def fit_modification2(self, X, y):
        X, y, max_iter = self.__initialize_fitting(X, y)
        iteration = 0
        num_changed = 0
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(y.shape[0]):
                    num_changed += self.__examine_example(i, X, y)
            else:
                while True:
                    i2 = self.__i_low
                    i1 = self.__i_up
                    inner_loop_success = self.__take_step(i1, i2, X, y)
                    num_changed += int(inner_loop_success)
                    if self.__b_up > self.__b_low - 2 * self.tol or not inner_loop_success:
                        break
                num_changed = 0
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            iteration += 1
            if iteration >= max_iter:
                break
        self.__set_result(X, y)

    def decision_function(self, X):
        if self.kernel == 'linear':
            return self.__coef.dot(X.T).ravel() - self.__threshold
        return np.sum(self.__dual_coef * self.__kernel_func(X, self.support_vectors), axis=1) - self.__threshold

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        y = np.asarray(y)
        return np.sum(self.predict(X) == y) / y.shape[0]
