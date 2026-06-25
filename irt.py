import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from scipy.integrate import quad
from scipy.special import hermite
import numpy.polynomial.hermite as herm

# Hàm tính độ phân biệt bằng point-biserial correlation
def cal_disc(r):
    r = np.nan_to_num(r, nan=1e-6) 
    return r/np.sqrt(1-r**2)
    
# Tính độ khó
def cal_diff(p) -> float:
    if p <= 0:
        p = 1e-6
    elif p >= 1:
        p = 1 - 1e-6
    
    return np.log((1-p)/p).clip(-6, 6)

# Hàm xác suất đúng theo mô hình IRT 2PL
def irt_probability(theta, a, b):
    theta = np.atleast_2d(theta).reshape(-1, 1)  # (K,1)
    a = np.asarray(a).reshape(1, -1)  # (1,J)
    b = np.asarray(b).reshape(1, -1)  # (1,J)
    z = 1.702 * a * (theta - b)
    return 1 / (1 + np.exp(-z))

def neg_log_likelihood(theta, responses, item_params):
    ll = 0.0
    for j, u in enumerate(responses):
        a, b = item_params[j]
        p = irt_probability(theta, a, b)
        if u == 1:
            ll += np.log(p + 1e-9)
        else:
            ll += np.log(1 - p + 1e-9)
    return -ll

def log_likelihood(U, a_list, b_list, theta_grid, gh_weights, eps=1e-12):
    """Compute the log-likelihood of the data given item parameters using Gauss-Hermite quadrature."""
    N, J = U.shape
    K = len(theta_grid)
    P_kj = irt_probability(theta_grid, a_list, b_list)  # (K, J)

    # đảm bảo không có 0 hoặc 1 tuyệt đối
    P_kj = np.clip(P_kj, eps, 1 - eps)

    logP = np.log(P_kj)
    log1mP = np.log(1.0 - P_kj)

    ll = 0.0
    mask = (U != -1)

    for i in range(N):
        u = U[i, :]
        m = mask[i, :]

        # chỉ tính những câu hợp lệ
        valid = (u != -1)
        u_valid = u[valid]
        logP_valid = logP[:, valid]
        log1mP_valid = log1mP[:, valid]

        term_k = u_valid @ logP_valid.T + (1 - u_valid) @ log1mP_valid.T
        term_k += np.log(gh_weights + eps)

        # nếu toàn bộ invalid → bỏ qua
        if np.all(~m):
            continue  

        # chống -inf toàn bộ
        if np.all(np.isneginf(term_k)):
            continue

        ll += logsumexp(term_k)

    return ll

def mmle(U, name="MMLE", max_iter=60, K=81, tol=1e-4, reg=1e-2, step_size=0.3, verbose=True):
    N, J = U.shape
    a = np.ones(J, dtype=float)
    b = np.zeros(J, dtype=float)

    # Gauss-Hermite nodes
    theta_grid, gh_weights = np.polynomial.hermite.hermgauss(K)
    theta_grid = theta_grid * np.sqrt(2)         # đúng chuẩn GH
    gh_weights = gh_weights / np.sqrt(np.pi)

    prev_ll = -np.inf
    mask = (U != -1)

    if verbose:
        print(f"Start {name}: N={N}, J={J}, K={K}")

    for it in range(1, max_iter + 1):
        # --- E-step ---
        P_kj = irt_probability(theta_grid, a, b)
        P_kj = np.clip(P_kj, 1e-12, 1-1e-12)
        logP = np.log(P_kj)
        log1mP = np.log1p(-P_kj)

        L = np.zeros((N, K))
        for k in range(K):
            L[:, k] = (mask * U) @ logP[k, :].T + (mask * (1 - U)) @ log1mP[k, :].T
            L[:, k] += np.log(gh_weights[k] + 1e-12)

        denom = logsumexp(L, axis=1)
        W = np.exp(L - denom[:, None])

        if verbose:
            print(f"Iter {it}: LL = {np.sum(denom):.2f}")

        a_old, b_old = a.copy(), b.copy()

        # --- M-step ---
        theta_k = theta_grid.reshape(K, 1)
        for j in range(J):
            col = U[:, j]
            valid_idx = np.where(col != -1)[0]
            if valid_idx.size == 0:
                continue
            prop = np.mean(col[valid_idx])
            # Thay continue bằng điều chỉnh P để tránh log(0), không bỏ item
            if prop < 0.01:
                prop = 0.01
            elif prop > 0.99:
                prop = 0.99

            u_vec = col[:, None]
            mask_vec = mask[:, j][:, None]
            P_kj_vec = P_kj[:, j].reshape(1, K)

            for _inner in range(2):
                D = (u_vec @ np.ones((1, K))) - P_kj_vec  # (N, K)
                D = D * mask_vec                          # mask theo rows
                term_theta = theta_k - b[j]

                grad_a = np.sum(W * D * (1.702 * term_theta).T) - reg * (a[j] - 1.0)
                grad_b = np.sum(W * D * (-a[j] * 1.702)) - reg * b[j]

                q_k = (P_kj_vec.flatten() * (1 - P_kj_vec.flatten()))  # (K,)

                q = q_k.reshape(1, K)
                tt = term_theta.flatten().reshape(1, K)

                hess_aa = -np.sum(W * (q * (1.702 * tt)**2)) - 1e-5
                hess_ab = -np.sum(W * (q * (1.702 * tt) * (1.702 * a[j])))
                hess_bb = -np.sum(W * (q * (1.702 * a[j])**2)) - 1e-5

                I = -np.array([[hess_aa, hess_ab], [hess_ab, hess_bb]])
                I_reg = I + reg * np.eye(2)
                g = np.array([grad_a, grad_b])

                try:
                    delta = np.linalg.solve(I_reg, g)
                except np.linalg.LinAlgError:
                    delta = 1e-3 * g

                max_step_a = 0.15
                max_step_b = 0.30

                delta[0] = np.clip(delta[0], -max_step_a, max_step_a)
                delta[1] = np.clip(delta[1], -max_step_b, max_step_b)

                a[j] = np.clip(a[j] + step_size * delta[0], 1e-3, 3.0)
                b[j] = np.clip(b[j] + step_size * delta[1], -6.0, 6.0)

        # --- check LL & convergence ---
        new_ll = log_likelihood(U, a, b, theta_grid, gh_weights)

        if new_ll < prev_ll - 1e-6:
            a, b = a_old, b_old
            step_size *= 0.7
            if step_size < 1e-4:
                break
            continue

        if (abs(new_ll - prev_ll) < tol) and (np.max(np.abs(a - a_old)) < tol) and (np.max(np.abs(b - b_old)) < tol):
            break

        prev_ll = new_ll

    return a, b

def theta_estimate(responses, item_params):
    theta_estimates = []
    for student_responses in responses:
        result = minimize(
            neg_log_likelihood,
            x0=0,
            args=(student_responses, item_params),
            bounds=[(-6, 6)],
            method="L-BFGS-B"
        )
        theta_estimates.append(float(result.x))
    return theta_estimates

def posterior(theta_grid, responses, item_params, prior_mean=0, prior_std=1, eps=1e-12):
    theta_grid = np.asarray(theta_grid)
    M = theta_grid.size

    prior = norm.pdf(theta_grid, loc=prior_mean, scale=prior_std)
    prior_sum = np.trapz(prior, theta_grid)
    if prior_sum <= 0:
        prior = np.ones_like(prior)
    else:
        prior = prior / prior_sum

    lik = np.ones_like(theta_grid, dtype=float) 
    item_params = np.asarray(item_params)
    J = item_params.shape[0]
    for j in range(J):
        a_j = float(item_params[j, 0])
        b_j = float(item_params[j, 1])
        p_j = 1.0 / (1.0 + np.exp(-1.702 * a_j * (theta_grid - b_j)))
        r = responses[j]
        if r == -1:
            continue
        lik *= (p_j ** (r)) * ((1.0 - p_j) ** (1 - r))

    post_unnorm = prior * lik
    integral = np.trapz(post_unnorm, theta_grid)
    if integral <= 0:
        post = prior.copy()
        post /= np.trapz(post, theta_grid)
    else:
        post = post_unnorm / integral

    return post

def ability_se(responses, item_params, theta_estimate, prior_mean=0, prior_std=1,
               theta_min=-6, theta_max=6, num_points=1001):
    theta_grid = np.linspace(theta_min, theta_max, num_points)
    post = posterior(theta_grid, responses, item_params, prior_mean, prior_std)
    post /= np.trapz(post, theta_grid)

    var = np.trapz((theta_grid - theta_estimate)**2 * post, theta_grid)
    if var < 0 and var > -1e-12:
        var = 0.0
    return np.sqrt(var)

def all_ability_se(responses_matrix, item_params, theta_estimates, prior_mean=0, prior_std=1,
                   theta_min=-6, theta_max=6, num_points=1001):
    ses = []
    for responses, theta_est in zip(responses_matrix, theta_estimates):
        se = ability_se(responses, item_params, theta_est, prior_mean, prior_std,
                        theta_min, theta_max, num_points)
        ses.append(se)
    return np.array(ses)

def item_se(a, b, prior_mean=0, prior_std=1, theta_min=-6, theta_max=6, num_quad=50, reg=1e-6):
    nodes, weights = herm.hermgauss(num_quad)
    theta_grid = prior_mean + np.sqrt(2) * prior_std * nodes
    prior = weights / np.sqrt(np.pi)

    p = 1.0 / (1.0 + np.exp(-1.702 * a * (theta_grid - b)))
    q = 1.0 - p

    dp_da = 1.702 * (theta_grid - b) * p * q
    dp_db = -1.702 * a * p * q

    with np.errstate(divide='ignore', invalid='ignore'):
        I_aa_theta = (dp_da ** 2) / (p * q + 1e-300)
        I_ab_theta = (dp_da * dp_db) / (p * q + 1e-300)
        I_bb_theta = (dp_db ** 2) / (p * q + 1e-300)

    I_aa = np.sum(I_aa_theta * prior)
    I_ab = np.sum(I_ab_theta * prior)
    I_bb = np.sum(I_bb_theta * prior)

    fisher = np.array([[I_aa, I_ab], [I_ab, I_bb]])
    fisher_reg = fisher + reg * np.eye(2)

    try:
        cov = np.linalg.inv(fisher_reg)
        diag = np.diag(cov).copy()
        diag[diag < 0] = np.inf
        se_a = np.sqrt(diag[0]) if np.isfinite(diag[0]) else np.inf
        se_b = np.sqrt(diag[1]) if np.isfinite(diag[1]) else np.inf
    except np.linalg.LinAlgError:
        se_a, se_b = np.inf, np.inf

    return se_a, se_b

def all_item_se(item_params, prior_mean=0, prior_std=1,
                theta_min=-6, theta_max=6, num_quad=50, reg=1e-6):
    ses = []
    for a, b in item_params:
        se_a, se_b = item_se(a, b, prior_mean=prior_mean, prior_std=prior_std,
                             theta_min=theta_min, theta_max=theta_max,
                             num_quad=num_quad, reg=reg)
        ses.append((se_a, se_b))
    return np.array(ses)

def chi_square(df, item_param, num_bins=12):
    theta = df["Theta"].to_numpy()
    item_cols = [col for col in df.columns if col in item_param.index]
    results = []

    def irt_prob(theta, a, b):
        return 1 / (1 + np.exp(-1.702 * a * (theta - b)))

    bins = np.linspace(-6, 6, num_bins + 1)
    bin_idx = np.digitize(theta, bins) - 1

    for item in item_cols:
        responses = df[item].replace(-1, 0).to_numpy()

        a = item_param.loc[item, "a"]
        b = item_param.loc[item, "b"]

        p = irt_prob(theta, a, b)
        q = 1 - p
        observed = np.zeros((num_bins, 2))
        expected = np.zeros((num_bins, 2))

        for i in range(num_bins):
            in_bin = (bin_idx == i)
            n_bin = np.sum(in_bin)

            if n_bin > 0:
                resp_bin = responses[in_bin]
                observed[i, 0] = np.sum(resp_bin == 1)
                observed[i, 1] = np.sum(resp_bin == 0)

                n_valid = np.sum((resp_bin == 1) | (resp_bin == 0))
                mask_valid = (resp_bin != -1)
                if mask_valid.sum() > 0:
                    expected[i, 0] = n_valid * p[in_bin][mask_valid].mean()
                    expected[i, 1] = n_valid * q[in_bin][mask_valid].mean()
                else:
                    expected[i, 0] = 0
                    expected[i, 1] = 0

        valid = expected.sum(axis=1) >= 5
        observed_valid = observed[valid]
        expected_valid = expected[valid]

        if expected_valid.shape[0] == 0:
            results.append([item, a, b, np.nan, np.nan])
            continue

        chi2_stat = np.sum((observed_valid - expected_valid)**2 / (expected_valid + 1e-9))
        df_chi = observed_valid.shape[0] - 2
        p_value = 1 - chi2.cdf(chi2_stat, df_chi)

        results.append([
            chi2_stat,
            round(p_value, 4)
        ])
    return pd.DataFrame(results, columns=["Chi2", "p_value"])

def true_score(theta, raw, data: pd.Series, item_params: pd.DataFrame) -> int:
    converted_score = 0.0
    max_score = 0.0

    for cau in data.index:
        a = item_params.loc[cau, "a"]
        b = item_params.loc[cau, "b"]

        p = irt_probability(theta, a, b)
        
        if data[cau] == 1:
            converted_score += p

        max_score += p

    if max_score == 0 or np.isnan(max_score):
        return 0
    if (theta == -6 and raw==0):
        return 0
    elif (theta == 6 and raw==30):
        return 300
    return int(np.clip(np.round(max_score * 10, 0), 0, 300))
