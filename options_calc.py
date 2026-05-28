"""
美股期权策略分析 - 期权计算核心模块
实现 Black-Scholes 定价模型及 Greeks 计算
"""

import numpy as np
from scipy.stats import norm
from typing import List, Dict, Any


class BlackScholes:
    """Black-Scholes 期权定价模型"""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """计算 d1 参数"""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """计算 d2 参数"""
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        计算看涨期权理论价格
        C = S * N(d1) - K * e^(-rT) * N(d2)
        """
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        计算看跌期权理论价格
        P = K * e^(-rT) * N(-d2) - S * N(-d1)
        """
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """根据期权类型返回理论价格"""
        if option_type == "call":
            return BlackScholes.call_price(S, K, T, r, sigma)
        elif option_type == "put":
            return BlackScholes.put_price(S, K, T, r, sigma)
        else:
            raise ValueError(f"无效的期权类型: {option_type}, 请输入 'call' 或 'put'")


class Greeks:
    """期权 Greeks 计算"""

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Delta - 期权价格对标的资产价格的敏感度
        Call: N(d1)
        Put: N(d1) - 1
        """
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        if option_type == "call":
            return norm.cdf(d1)
        elif option_type == "put":
            return norm.cdf(d1) - 1.0
        else:
            raise ValueError(f"无效的期权类型: {option_type}")

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Gamma - Delta 对标的资产价格的敏感度
        Gamma = n(d1) / (S * sigma * sqrt(T))
        """
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Theta - 期权价格对时间的敏感度（年化）
        按交易日计算，除以365转换为每日
        """
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)

        common_term = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

        if option_type == "call":
            theta_annual = common_term - r * K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            theta_annual = common_term + r * K * np.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError(f"无效的期权类型: {option_type}")

        # 返回每日Theta（年化除以365）
        return theta_annual / 365.0

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Vega - 期权价格对波动率的敏感度
        Vega = S * sqrt(T) * n(d1)
        注意：返回的是每1%波动率变化的价格变动
        """
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * np.sqrt(T) * norm.pdf(d1) / 100.0

    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Rho - 期权价格对无风险利率的敏感度
        返回的是每1%利率变化的价格变动
        """
        d2 = BlackScholes.d2(S, K, T, r, sigma)

        if option_type == "call":
            rho_val = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
        elif option_type == "put":
            rho_val = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0
        else:
            raise ValueError(f"无效的期权类型: {option_type}")

        return rho_val

    @staticmethod
    def all(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, float]:
        """计算所有 Greeks 并返回字典"""
        return {
            "delta": Greeks.delta(S, K, T, r, sigma, option_type),
            "gamma": Greeks.gamma(S, K, T, r, sigma),
            "theta": Greeks.theta(S, K, T, r, sigma, option_type),
            "vega": Greeks.vega(S, K, T, r, sigma),
            "rho": Greeks.rho(S, K, T, r, sigma, option_type),
        }


class PayoffCalculator:
    """期权盈亏计算器"""

    @staticmethod
    def single_option_payoff(
        S_range: np.ndarray,
        K: float,
        premium: float,
        option_type: str,
        position: str = "long"
    ) -> np.ndarray:
        """
        计算单个期权到期盈亏
        S_range: 标的价格区间
        K: 行权价
        premium: 期权权利金（成本）
        option_type: call / put
        position: long / short
        """
        if option_type == "call":
            intrinsic = np.maximum(S_range - K, 0)
        elif option_type == "put":
            intrinsic = np.maximum(K - S_range, 0)
        else:
            raise ValueError(f"无效的期权类型: {option_type}")

        if position == "long":
            payoff = intrinsic - premium
        elif position == "short":
            payoff = premium - intrinsic
        else:
            raise ValueError(f"无效的持仓方向: {position}")

        return payoff

    @staticmethod
    def strategy_payoff(
        S_range: np.ndarray,
        legs: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        计算组合策略到期盈亏
        legs: 期权腿列表，每项包含:
            - K: 行权价
            - premium: 权利金
            - option_type: call / put
            - position: long / short
            - quantity: 数量（默认1）
        """
        total_payoff = np.zeros_like(S_range)

        for leg in legs:
            K = leg["K"]
            premium = leg["premium"]
            option_type = leg["option_type"]
            position = leg.get("position", "long")
            quantity = leg.get("quantity", 1)

            leg_payoff = PayoffCalculator.single_option_payoff(
                S_range, K, premium, option_type, position
            )
            total_payoff += leg_payoff * quantity

        return total_payoff

    @staticmethod
    def find_breakeven_points(S_range: np.ndarray, payoff: np.ndarray) -> List[float]:
        """查找盈亏平衡点"""
        breakevens = []
        for i in range(len(payoff) - 1):
            if payoff[i] * payoff[i + 1] < 0:
                # 线性插值
                x0, x1 = S_range[i], S_range[i + 1]
                y0, y1 = payoff[i], payoff[i + 1]
                be = x0 - y0 * (x1 - x0) / (y1 - y0)
                breakevens.append(round(be, 2))
        return breakevens

    @staticmethod
    def max_profit_loss(payoff: np.ndarray) -> Dict[str, float]:
        """计算最大收益和最大亏损"""
        return {
            "max_profit": round(float(np.max(payoff)), 2),
            "max_loss": round(float(np.min(payoff)), 2),
        }


def calculate_full_analysis(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    num_points: int = 200
) -> Dict[str, Any]:
    """
    完整期权分析：价格、Greeks、盈亏数据

    参数：
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率（小数形式）
        option_type: call / put
        num_points: 盈亏图采样点数

    返回：
        包含所有分析结果的字典
    """
    # 基本校验
    if S <= 0 or K <= 0 or sigma <= 0:
        raise ValueError("标的价格、行权价、波动率必须大于0")
    if T < 0:
        raise ValueError("到期时间不能为负")
    if T == 0:
        # 到期日：只有内在价值
        if option_type == "call":
            price = max(S - K, 0)
        else:
            price = max(K - S, 0)
        return {
            "price": round(price, 4),
            "greeks": {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0},
        }

    # 计算理论价格
    price = BlackScholes.price(S, K, T, r, sigma, option_type)

    # 计算 Greeks
    greeks = Greeks.all(S, K, T, r, sigma, option_type)

    return {
        "price": round(price, 4),
        "greeks": {k: round(v, 6) for k, v in greeks.items()},
    }


def calculate_payoff_data(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    num_points: int = 200
) -> Dict[str, Any]:
    """
    计算盈亏图数据

    返回：
        - prices: 价格区间
        - payoff: 盈亏数据
        - breakeven: 盈亏平衡点
        - max_profit: 最大收益
        - max_loss: 最大亏损
        - current_price: 当前标的价格
        - premium: 期权权利金
    """
    # 计算权利金
    premium = BlackScholes.price(S, K, T, r, sigma, option_type)

    # 生成价格区间（行权价上下各50%）
    lower = max(K * 0.5, S * 0.5)
    upper = K * 1.5 + S * 0.5
    S_range = np.linspace(lower, upper, num_points)

    # 计算盈亏
    payoff = PayoffCalculator.single_option_payoff(S_range, K, premium, option_type, "long")

    # 盈亏平衡点
    breakevens = PayoffCalculator.find_breakeven_points(S_range, payoff)

    # 最大收益/亏损
    pl = PayoffCalculator.max_profit_loss(payoff)

    return {
        "prices": [round(float(x), 2) for x in S_range],
        "payoff": [round(float(x), 2) for x in payoff],
        "breakeven": breakevens,
        "max_profit": pl["max_profit"],
        "max_loss": pl["max_loss"],
        "current_price": S,
        "premium": round(premium, 4),
    }


def calculate_strategy_payoff_data(
    S: float,
    legs: List[Dict[str, Any]],
    T: float,
    r: float,
    sigma: float,
    num_points: int = 200
) -> Dict[str, Any]:
    """
    计算组合策略盈亏图数据

    参数：
        S: 当前标的价格
        legs: 策略腿列表
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        num_points: 采样点数
    """
    # 计算每腿的权利金
    computed_legs = []
    for leg in legs:
        K = leg["K"]
        option_type = leg["option_type"]
        position = leg.get("position", "long")
        quantity = leg.get("quantity", 1)

        premium = BlackScholes.price(S, K, T, r, sigma, option_type)
        computed_legs.append({
            "K": K,
            "premium": premium,
            "option_type": option_type,
            "position": position,
            "quantity": quantity,
        })

    # 生成价格区间
    all_K = [leg["K"] for leg in legs]
    lower = max(min(all_K) * 0.5, S * 0.5)
    upper = max(all_K) * 1.5 + S * 0.3
    S_range = np.linspace(lower, upper, num_points)

    # 计算组合盈亏
    payoff = PayoffCalculator.strategy_payoff(S_range, computed_legs)

    # 盈亏平衡点
    breakevens = PayoffCalculator.find_breakeven_points(S_range, payoff)

    # 最大收益/亏损
    pl = PayoffCalculator.max_profit_loss(payoff)

    return {
        "prices": [round(float(x), 2) for x in S_range],
        "payoff": [round(float(x), 2) for x in payoff],
        "breakeven": breakevens,
        "max_profit": pl["max_profit"],
        "max_loss": pl["max_loss"],
        "current_price": S,
        "legs": [{
            "K": leg["K"],
            "premium": round(leg["premium"], 4),
            "option_type": leg["option_type"],
            "position": leg["position"],
            "quantity": leg.get("quantity", 1),
        } for leg in computed_legs],
    }
