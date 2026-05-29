#!/usr/bin/env python3
"""
数学建模常用 Python 算法工具箱
覆盖：优化、统计、微分方程、图论、模拟等
"""

import numpy as np
from scipy import optimize, integrate, interpolate, linalg
from scipy.stats import norm, t, f, chi2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 线性规划 (Linear Programming)
# ============================================================
def linear_programming_example():
    """
    max z = 2x1 + 3x2
    s.t.
        x1 + 2x2 <= 8
        4x1      <= 16
            4x2  <= 12
        x1, x2 >= 0
    """
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value

    prob = LpProblem("LP_Example", LpMaximize)
    x1 = LpVariable("x1", lowBound=0)
    x2 = LpVariable("x2", lowBound=0)

    prob += 2 * x1 + 3 * x2
    prob += x1 + 2 * x2 <= 8
    prob += 4 * x1 <= 16
    prob += 4 * x2 <= 12

    prob.solve()
    print(f"状态: {LpStatus[prob.status]}")
    print(f"x1 = {value(x1)}, x2 = {value(x2)}, 最优值 = {value(prob.objective)}")
    return value(x1), value(x2), value(prob.objective)


# ============================================================
# 2. 非线性优化 (Nonlinear Optimization)
# ============================================================
def nonlinear_optimization_example():
    """min f(x,y) = (x-1)^2 + (y-2.5)^2, 无约束"""
    result = optimize.minimize(
        lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2,
        x0=[0, 0],
        method="BFGS",
    )
    print(f"最优解: {result.x}, 最优值: {result.fun}")
    return result.x, result.fun


def constrained_optimization_example():
    """
    min f(x,y) = (x-1)^2 + (y-2.5)^2
    s.t.
        x + y >= 1
        x >= 0, y >= 0
    """
    constraints = [{"type": "ineq", "fun": lambda x: x[0] + x[1] - 1}]
    bounds = optimize.Bounds([0, 0], [np.inf, np.inf])
    result = optimize.minimize(
        lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2,
        x0=[0.5, 0.5],
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )
    print(f"最优解: {result.x}, 最优值: {result.fun}")
    return result.x, result.fun


# ============================================================
# 3. 插值 & 拟合 (Interpolation & Curve Fitting)
# ============================================================
def interpolation_example():
    """三次样条插值"""
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 2, 1, 3, 8, 10])
    cs = interpolate.CubicSpline(x, y)
    x_new = np.linspace(0, 5, 100)
    y_new = cs(x_new)

    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, c="red", label="Original Data")
    plt.plot(x_new, y_new, label="Cubic Spline")
    plt.legend()
    plt.title("Cubic Spline Interpolation")
    plt.savefig("interpolation.png", dpi=150)
    plt.close()
    print("插值图已保存为 interpolation.png")
    return cs


def curve_fitting_example():
    """最小二乘拟合 y = a*exp(b*x) + c"""
    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    y = np.array([1.8, 2.6, 4.1, 6.8, 10.2])

    def model(x, a, b, c):
        return a * np.exp(b * x) + c

    params, cov = optimize.curve_fit(model, x, y, p0=[1, 0.5, 0])
    a, b, c = params
    print(f"拟合参数: a={a:.4f}, b={b:.4f}, c={c:.4f}")

    x_plot = np.linspace(0.4, 2.6, 100)
    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, c="red", label="Data Points")
    plt.plot(x_plot, model(x_plot, *params), label="Fitted Curve")
    plt.legend()
    plt.title("Curve Fitting: a*exp(b*x) + c")
    plt.savefig("curve_fitting.png", dpi=150)
    plt.close()
    return params, cov


# ============================================================
# 4. 常微分方程 (ODE)
# ============================================================
def ode_example():
    """Lotka-Volterra 捕食者-猎物模型
    dx/dt = ax - bxy
    dy/dt = cxy - dy
    """
    def lotka_volterra(t, z, a, b, c, d):
        x, y = z
        return [a * x - b * x * y, c * x * y - d * y]

    a, b, c, d = 0.5, 0.3, 0.2, 0.6
    t_span = (0, 50)
    t_eval = np.linspace(0, 50, 500)
    sol = integrate.solve_ivp(
        lotka_volterra, t_span, [2, 1], args=(a, b, c, d), t_eval=t_eval
    )

    plt.figure(figsize=(8, 4))
    plt.plot(sol.t, sol.y[0], label="Prey")
    plt.plot(sol.t, sol.y[1], label="Predator")
    plt.xlabel("t")
    plt.ylabel("Population")
    plt.legend()
    plt.title("Lotka-Volterra Model")
    plt.savefig("ode_lotka_volterra.png", dpi=150)
    plt.close()
    print("ODE 图已保存为 ode_lotka_volterra.png")
    return sol


# ============================================================
# 5. 蒙特卡洛模拟 (Monte Carlo Simulation)
# ============================================================
def monte_carlo_pi(n=100000):
    """用蒙特卡洛方法估算 pi"""
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    inside = x**2 + y**2 <= 1
    pi_approx = 4 * inside.sum() / n
    print(f"蒙特卡洛估算 pi ≈ {pi_approx:.6f} (误差: {abs(pi_approx - np.pi):.6f})")
    return pi_approx


def monte_carlo_integration(n=100000):
    """蒙特卡洛积分: ∫_0^1 sin(x^2) dx"""
    x = np.random.uniform(0, 1, n)
    estimates = np.sin(x**2)
    integral = estimates.mean()
    # 真实值
    from scipy.special import fresnel

    true_val, _ = fresnel(np.sqrt(2 / np.pi))
    true_val *= np.sqrt(np.pi / 2)
    print(
        f"蒙特卡洛积分 ≈ {integral:.6f} (真实值 ≈ {true_val:.6f}, 误差: {abs(integral - true_val):.6f})"
    )
    return integral


# ============================================================
# 6. 主成分分析 (PCA)
# ============================================================
def pca_example():
    """对随机数据执行 PCA 降维"""
    from sklearn.decomposition import PCA

    np.random.seed(42)
    X = np.dot(np.random.randn(100, 3), np.random.randn(3, 3)) + np.array([1, 2, 3])

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    print(f"原始维度: {X.shape}, 降维后: {X_reduced.shape}")
    print(f"各主成分方差解释比例: {pca.explained_variance_ratio_}")
    print(f"累计方差解释比例: {pca.explained_variance_ratio_.cumsum()}")

    plt.figure(figsize=(8, 4))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D Projection")
    plt.savefig("pca.png", dpi=150)
    plt.close()
    return X_reduced, pca


# ============================================================
# 7. 图论 (Graph Theory) — NetworkX
# ============================================================
def graph_example():
    """最短路径、最小生成树"""
    import networkx as nx

    G = nx.Graph()
    edges = [
        ("A", "B", 4),
        ("A", "C", 2),
        ("B", "C", 1),
        ("B", "D", 5),
        ("C", "D", 8),
        ("C", "E", 10),
        ("D", "E", 2),
        ("D", "F", 6),
        ("E", "F", 3),
    ]
    G.add_weighted_edges_from(edges)

    shortest = nx.dijkstra_path(G, "A", "F", weight="weight")
    shortest_len = nx.dijkstra_path_length(G, "A", "F", weight="weight")
    print(f"最短路径 A→F: {shortest}, 长度: {shortest_len}")

    mst = nx.minimum_spanning_tree(G, weight="weight")
    print(f"最小生成树边: {list(mst.edges(data=True))}")

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=600)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Graph: Shortest Path & MST")
    plt.savefig("graph.png", dpi=150)
    plt.close()
    return G, shortest, mst


# ============================================================
# 8. 假设检验 (Hypothesis Testing)
# ============================================================
def hypothesis_testing_example():
    """独立样本 t 检验"""
    np.random.seed(42)
    group_a = np.random.normal(10, 2, 30)
    group_b = np.random.normal(11, 2, 30)

    from scipy.stats import ttest_ind

    t_stat, p_value = ttest_ind(group_a, group_b)
    print(f"t 统计量: {t_stat:.4f}, p 值: {p_value:.4f}")
    if p_value < 0.05:
        print("拒绝原假设，两组均值存在显著差异")
    else:
        print("不能拒绝原假设")
    return t_stat, p_value


# ============================================================
# 9. 回归分析 (Regression)
# ============================================================
def linear_regression_example():
    """多元线性回归"""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    np.random.seed(0)
    n = 200
    X = np.random.randn(n, 3)
    true_beta = np.array([2.5, -1.2, 0.8])
    y = X @ true_beta + 3.0 + np.random.randn(n) * 0.5

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"系数: {model.coef_}, 截距: {model.intercept_:.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    return model


# ============================================================
# 10. 聚类 (Clustering)
# ============================================================
def clustering_example():
    """K-Means 聚类"""
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=0)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        c="red",
        marker="x",
        s=200,
        linewidths=3,
    )
    plt.title("K-Means Clustering")
    plt.savefig("clustering.png", dpi=150)
    plt.close()
    print(f"聚类中心:\n{kmeans.cluster_centers_}")
    return kmeans, labels


# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("1. 线性规划")
    linear_programming_example()

    print("\n" + "=" * 50)
    print("2. 非线性优化")
    nonlinear_optimization_example()
    constrained_optimization_example()

    print("\n" + "=" * 50)
    print("3. 插值与拟合")
    interpolation_example()
    curve_fitting_example()

    print("\n" + "=" * 50)
    print("4. 常微分方程")
    ode_example()

    print("\n" + "=" * 50)
    print("5. 蒙特卡洛模拟")
    monte_carlo_pi(200000)
    monte_carlo_integration(200000)

    print("\n" + "=" * 50)
    print("6. PCA 降维")
    pca_example()

    print("\n" + "=" * 50)
    print("7. 图论")
    graph_example()

    print("\n" + "=" * 50)
    print("8. 假设检验")
    hypothesis_testing_example()

    print("\n" + "=" * 50)
    print("9. 线性回归")
    linear_regression_example()

    print("\n" + "=" * 50)
    print("10. 聚类")
    clustering_example()

    print("\n所有算法运行完毕！")
