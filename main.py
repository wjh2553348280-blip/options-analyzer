"""
美股期权策略分析 - FastAPI 后端
提供期权定价、Greeks计算、盈亏分析等 REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os

from options_calc import (
    BlackScholes,
    Greeks,
    PayoffCalculator,
    calculate_full_analysis,
    calculate_payoff_data,
    calculate_strategy_payoff_data,
)

# ============================================================
# FastAPI 应用实例
# ============================================================
app = FastAPI(
    title="美股期权策略分析",
    description="基于 Black-Scholes 模型的期权定价与 Greeks 分析系统",
    version="1.0.0",
)

# 挂载静态文件
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ============================================================
# 请求模型定义
# ============================================================
class OptionRequest(BaseModel):
    """单期权计算请求"""
    S: float = Field(..., gt=0, description="标的价格")
    K: float = Field(..., gt=0, description="行权价")
    T: float = Field(..., ge=0, description="到期时间（天数）")
    r: float = Field(0.045, ge=0, le=0.2, description="无风险利率（小数形式）")
    sigma: float = Field(..., gt=0, le=5.0, description="波动率 IV（小数形式）")
    option_type: str = Field(..., pattern="^(call|put)$", description="期权类型: call / put")


class OptionGreeksRequest(BaseModel):
    """Greeks 计算请求"""
    S: float = Field(..., gt=0, description="标的价格")
    K: float = Field(..., gt=0, description="行权价")
    T: float = Field(..., ge=0, description="到期时间（天数）")
    r: float = Field(0.045, ge=0, le=0.2, description="无风险利率")
    sigma: float = Field(..., gt=0, le=5.0, description="波动率 IV")
    option_type: str = Field(..., pattern="^(call|put)$", description="期权类型")


class PayoffRequest(BaseModel):
    """盈亏图数据请求"""
    S: float = Field(..., gt=0, description="标的价格")
    K: float = Field(..., gt=0, description="行权价")
    T: float = Field(..., ge=0, description="到期时间（天数）")
    r: float = Field(0.045, ge=0, le=0.2, description="无风险利率")
    sigma: float = Field(..., gt=0, le=5.0, description="波动率 IV")
    option_type: str = Field(..., pattern="^(call|put)$", description="期权类型")
    num_points: int = Field(200, ge=50, le=500, description="采样点数")


class StrategyLeg(BaseModel):
    """策略腿定义"""
    K: float = Field(..., gt=0, description="行权价")
    option_type: str = Field(..., pattern="^(call|put)$", description="期权类型")
    position: str = Field("long", pattern="^(long|short)$", description="持仓方向")
    quantity: int = Field(1, ge=1, le=100, description="数量")


class StrategyPayoffRequest(BaseModel):
    """策略盈亏图请求"""
    S: float = Field(..., gt=0, description="标的价格")
    T: float = Field(..., ge=0, description="到期时间（天数）")
    r: float = Field(0.045, ge=0, le=0.2, description="无风险利率")
    sigma: float = Field(..., gt=0, le=5.0, description="波动率 IV")
    legs: List[StrategyLeg] = Field(..., min_length=1, max_length=10, description="策略腿列表")
    num_points: int = Field(200, ge=50, le=500, description="采样点数")


class IVSensitivityRequest(BaseModel):
    """IV 敏感性分析请求"""
    S: float = Field(..., gt=0, description="标的价格")
    K: float = Field(..., gt=0, description="行权价")
    T: float = Field(..., ge=0, description="到期时间（天数）")
    r: float = Field(0.045, ge=0, le=0.2, description="无风险利率")
    sigma_min: float = Field(0.05, gt=0, description="最小波动率")
    sigma_max: float = Field(1.0, gt=0, description="最大波动率")
    option_type: str = Field(..., pattern="^(call|put)$", description="期权类型")
    num_points: int = Field(50, ge=10, le=200, description="采样点数")


# ============================================================
# API 路由
# ============================================================

@app.get("/", include_in_schema=False)
async def root():
    """返回前端页面"""
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "message": "期权分析系统运行正常"}


@app.post("/calculate")
async def calculate_option(req: OptionRequest):
    """
    计算期权理论价格与 Greeks

    返回：
        - price: 期权理论价格
        - greeks: Delta, Gamma, Theta, Vega, Rho
    """
    try:
        T = req.T / 365.0  # 天数转年
        result = calculate_full_analysis(
            S=req.S, K=req.K, T=T, r=req.r,
            sigma=req.sigma, option_type=req.option_type
        )
        return {
            "success": True,
            "data": {
                "price": result["price"],
                "greeks": result["greeks"],
                "input": {
                    "S": req.S, "K": req.K, "T_days": req.T,
                    "r": req.r, "sigma": req.sigma,
                    "option_type": req.option_type,
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"计算错误: {str(e)}")


@app.post("/greeks")
async def calculate_greeks(req: OptionGreeksRequest):
    """
    单独计算 Greeks

    返回：Delta, Gamma, Theta, Vega, Rho
    """
    try:
        T = req.T / 365.0
        if T == 0:
            return {
                "success": True,
                "data": {
                    "delta": 0, "gamma": 0, "theta": 0,
                    "vega": 0, "rho": 0
                }
            }

        greeks = Greeks.all(req.S, req.K, T, req.r, req.sigma, req.option_type)
        return {
            "success": True,
            "data": {k: round(v, 6) for k, v in greeks.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Greeks 计算错误: {str(e)}")


@app.post("/payoff")
async def calculate_payoff(req: PayoffRequest):
    """
    计算盈亏图数据

    返回：
        - prices: 价格区间
        - payoff: 盈亏数据
        - breakeven: 盈亏平衡点
        - max_profit: 最大收益
        - max_loss: 最大亏损
    """
    try:
        T = req.T / 365.0
        result = calculate_payoff_data(
            S=req.S, K=req.K, T=T, r=req.r,
            sigma=req.sigma, option_type=req.option_type,
            num_points=req.num_points
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"盈亏计算错误: {str(e)}")


@app.post("/strategy")
async def calculate_strategy(req: StrategyPayoffRequest):
    """
    计算组合策略盈亏图

    支持多腿策略：Covered Call, Bull Spread, Iron Condor 等
    """
    try:
        T = req.T / 365.0
        legs = [leg.model_dump() for leg in req.legs]
        result = calculate_strategy_payoff_data(
            S=req.S, legs=legs, T=T, r=req.r,
            sigma=req.sigma, num_points=req.num_points
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"策略计算错误: {str(e)}")


@app.post("/iv-sensitivity")
async def iv_sensitivity(req: IVSensitivityRequest):
    """
    波动率敏感性分析

    返回不同 IV 水平下的：
        - 理论价格
        - Greeks 值
    """
    try:
        T = req.T / 365.0
        if T == 0:
            raise HTTPException(status_code=400, detail="到期时间为0，无法进行IV敏感性分析")

        import numpy as np
        sigma_range = np.linspace(req.sigma_min, req.sigma_max, req.num_points)

        prices = []
        deltas = []
        gammas = []
        thetas = []
        vegas = []

        for sigma in sigma_range:
            price = BlackScholes.price(req.S, req.K, T, req.r, sigma, req.option_type)
            greeks = Greeks.all(req.S, req.K, T, req.r, sigma, req.option_type)

            prices.append(round(float(price), 4))
            deltas.append(round(float(greeks["delta"]), 6))
            gammas.append(round(float(greeks["gamma"]), 6))
            thetas.append(round(float(greeks["theta"]), 6))
            vegas.append(round(float(greeks["vega"]), 6))

        return {
            "success": True,
            "data": {
                "sigma_range": [round(float(x), 4) for x in sigma_range],
                "prices": prices,
                "deltas": deltas,
                "gammas": gammas,
                "thetas": thetas,
                "vegas": vegas,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"IV分析错误: {str(e)}")


# ============================================================
# 预设策略模板
# ============================================================

@app.get("/strategies")
async def list_strategies():
    """
    返回预设策略模板列表
    方便前端直接调用预设策略
    """
    strategies = {
        "long_call": {
            "name": "买入看涨 (Long Call)",
            "description": "看涨策略，最大亏损为权利金",
            "legs": [
                {"K_offset": 0, "option_type": "call", "position": "long", "quantity": 1}
            ]
        },
        "long_put": {
            "name": "买入看跌 (Long Put)",
            "description": "看跌策略，最大亏损为权利金",
            "legs": [
                {"K_offset": 0, "option_type": "put", "position": "long", "quantity": 1}
            ]
        },
        "covered_call": {
            "name": "备兑看涨 (Covered Call)",
            "description": "持有正股 + 卖出看涨期权",
            "legs": [
                {"K_offset": 0, "option_type": "call", "position": "short", "quantity": 1}
            ],
            "note": "需配合正股持仓，盈亏图仅显示期权部分"
        },
        "protective_put": {
            "name": "保护性看跌 (Protective Put)",
            "description": "持有正股 + 买入看跌期权",
            "legs": [
                {"K_offset": 0, "option_type": "put", "position": "long", "quantity": 1}
            ],
            "note": "需配合正股持仓，盈亏图仅显示期权部分"
        },
        "bull_call_spread": {
            "name": "牛市看涨价差 (Bull Call Spread)",
            "description": "买入低行权价Call + 卖出高行权价Call",
            "legs": [
                {"K_offset": -0.05, "option_type": "call", "position": "long", "quantity": 1},
                {"K_offset": 0.05, "option_type": "call", "position": "short", "quantity": 1}
            ]
        },
        "bear_put_spread": {
            "name": "熊市看跌价差 (Bear Put Spread)",
            "description": "买入高行权价Put + 卖出低行权价Put",
            "legs": [
                {"K_offset": 0.05, "option_type": "put", "position": "long", "quantity": 1},
                {"K_offset": -0.05, "option_type": "put", "position": "short", "quantity": 1}
            ]
        },
        "straddle": {
            "name": "跨式策略 (Straddle)",
            "description": "同时买入相同行权价的Call和Put",
            "legs": [
                {"K_offset": 0, "option_type": "call", "position": "long", "quantity": 1},
                {"K_offset": 0, "option_type": "put", "position": "long", "quantity": 1}
            ]
        },
        "strangle": {
            "name": "宽跨式策略 (Strangle)",
            "description": "买入不同行权价的Call和Put",
            "legs": [
                {"K_offset": 0.05, "option_type": "call", "position": "long", "quantity": 1},
                {"K_offset": -0.05, "option_type": "put", "position": "long", "quantity": 1}
            ]
        },
        "iron_condor": {
            "name": "铁鹰策略 (Iron Condor)",
            "description": "卖出中等行权价价差，买入远端保护",
            "legs": [
                {"K_offset": -0.10, "option_type": "put", "position": "long", "quantity": 1},
                {"K_offset": -0.05, "option_type": "put", "position": "short", "quantity": 1},
                {"K_offset": 0.05, "option_type": "call", "position": "short", "quantity": 1},
                {"K_offset": 0.10, "option_type": "call", "position": "long", "quantity": 1}
            ]
        },
    }
    return {"success": True, "data": strategies}


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
