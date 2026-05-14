import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import requests

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel

load_dotenv()

console = Console()

def _print_result(content) -> None:
    """安全打印 Agent 结果：清理 emoji 等 GBK 不支持的字符后输出"""
    text = str(content)
    # 过滤掉 GBK 无法编码的字符（如 emoji）
    text = text.encode('gbk', errors='ignore').decode('gbk')
    try:
        console.print(Panel(text, border_style="green", title="结果"))
    except UnicodeEncodeError:
        print(text)

model = ChatOpenAI(
    model="qwen3.5-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-777ae59d8b3e451db4dd91fe6961dbe5"
)

# ============================================================
# API 常量
# ============================================================
TOKEN = "zgaLG8unUPr"
BASE_URL = "https://api.autostock.cn/v1"


def _safe_get(url: str, params: dict = None, timeout: int = 10) -> Dict:
    """安全 GET 请求"""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def _safe_post(url: str, payload: dict, timeout: int = 10) -> Dict:
    """安全 POST 请求"""
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 工具函数 —— 股票数据查询
# ============================================================

# ---- 原始 API 函数（供 tool 和内部逻辑复用）----

def _search_stock(keyword: str) -> Dict:
    url = f"{BASE_URL}/stock/all?token={TOKEN}&keyWord={keyword}"
    return _safe_get(url)

def _get_stock_info(code: str) -> Dict:
    url = f"{BASE_URL}/stock?token={TOKEN}&code={code}"
    return _safe_get(url)

def _get_day_kline(code: str, startDate: Optional[str] = None,
                   endDate: Optional[str] = None, fq_type: int = 1) -> Dict:
    url = f"{BASE_URL}/stock/kline/day?token={TOKEN}"
    return _safe_get(url, {"code": code, "startDate": startDate, "endDate": endDate, "type": fq_type})

def _get_week_kline(code: str, startDate: Optional[str] = None,
                    endDate: Optional[str] = None, fq_type: int = 1) -> Dict:
    url = f"{BASE_URL}/stock/kline/week?token={TOKEN}"
    return _safe_get(url, {"code": code, "startDate": startDate, "endDate": endDate, "type": fq_type})

def _get_month_kline(code: str, startDate: Optional[str] = None,
                     endDate: Optional[str] = None, fq_type: int = 1) -> Dict:
    url = f"{BASE_URL}/stock/kline/month?token={TOKEN}"
    return _safe_get(url, {"code": code, "startDate": startDate, "endDate": endDate, "type": fq_type})

def _get_stock_rank(node: str = "a", sort: str = "priceChange", asc: int = 0,
                    pageIndex: int = 1, pageSize: int = 50) -> Dict:
    url = f"{BASE_URL}/stock/rank?token={TOKEN}"
    payload = {"node": node, "sort": sort, "asc": asc, "pageIndex": pageIndex, "pageSize": pageSize}
    return _safe_post(url, payload)

def _get_board_info() -> Dict:
    url = f"{BASE_URL}/stock/board?token={TOKEN}"
    return _safe_get(url)

def _get_stock_minute_data(code: str) -> Dict:
    url = f"{BASE_URL}/stock/min?token={TOKEN}&code={code}"
    return _safe_get(url)


# ---- @tool 装饰（给 Agent 调用的接口，不参与内部逻辑）----

@tool
def search_stock(keyword: str) -> Dict:
    """搜索股票，支持代码和名称模糊查询。如 search_stock('茅台') 或 search_stock('600519')"""
    return _search_stock(keyword)

@tool
def get_stock_info(code: str) -> Dict:
    """获取单只股票基础信息（公司概况、所属行业等）。code: 股票代码，如 '600519'"""
    return _get_stock_info(code)


@tool
def get_day_kline(code: str, startDate: Optional[str] = None,
                  endDate: Optional[str] = None, fq_type: int = 1) -> Dict:
    """获取日K线数据。code: 股票代码；startDate/endDate: '2025-01-01'格式；fq_type: 0不复权,1前复权(默认),2后复权"""
    return _get_day_kline(code, startDate, endDate, fq_type)

@tool
def get_week_kline(code: str, startDate: Optional[str] = None,
                   endDate: Optional[str] = None, fq_type: int = 1) -> Dict:
    """获取周K线数据。参数同 get_day_kline"""
    return _get_week_kline(code, startDate, endDate, fq_type)

@tool
def get_month_kline(code: str, startDate: Optional[str] = None,
                    endDate: Optional[str] = None, fq_type: int = 1) -> Dict:
    """获取月K线数据。用于判断长期趋势方向。参数同 get_day_kline"""
    return _get_month_kline(code, startDate, endDate, fq_type)

@tool
def get_stock_rank(node: str = "a", sort: str = "priceChange", asc: int = 0,
                   pageIndex: int = 1, pageSize: int = 50) -> Dict:
    """股票排行榜。node: 'a'(沪深A股),'b','ash','asz','bsh','bsz'；sort: price,priceChange,pricePercent,volume等；asc: 0降序,1升序"""
    return _get_stock_rank(node, sort, asc, pageIndex, pageSize)

@tool
def get_board_info() -> Dict:
    """获取大盘数据（上证指数、深证成指、创业板指等）"""
    return _get_board_info()

@tool
def get_stock_minute_data(code: str) -> Dict:
    """获取股票分时数据。code: 股票代码"""
    return _get_stock_minute_data(code)


# ============================================================
# 辅助函数 —— 数据解析与规范化
# ============================================================

def _parse_kline_records(raw: Dict) -> List[Dict]:
    """从 API 返回的原始数据中提取 K线记录列表"""
    if not raw or "error" in raw:
        return []
    data = raw.get("data", raw)
    if isinstance(data, dict):
        records = data.get("records", data.get("list", data.get("items", [])))
    elif isinstance(data, list):
        records = data
    else:
        return []
    if not records:
        return []
    # 如果记录是 list 而非 dict（API 返回 [date, open, close, high, low, volume] 格式），
    # 转换为带键名的 dict，以免后续 DataFrame 列名变成数字索引
    if isinstance(records[0], list):
        keys = ["date", "open", "close", "high", "low", "volume"]
        records = [dict(zip(keys, r)) for r in records]
    return records


def _normalize_kline_df(records: List[Dict]) -> pd.DataFrame:
    """将原始 K线记录转为规范化 DataFrame，列名统一为 date/open/close/high/low/volume"""
    df = pd.DataFrame(records)
    col_map = {}
    for c in df.columns:
        if not isinstance(c, str):
            continue
        cl = c.lower()
        if 'date' in cl or 'time' in cl or cl == 'day' or cl == 'trade_date':
            col_map[c] = 'date'
        elif cl in ('open', 'close', 'high', 'low', 'volume'):
            col_map[c] = cl
        elif 'open' in cl:
            col_map[c] = 'open'
        elif 'close' in cl:
            col_map[c] = 'close'
        elif 'high' in cl:
            col_map[c] = 'high'
        elif 'low' in cl:
            col_map[c] = 'low'
        elif 'vol' in cl:
            col_map[c] = 'volume'
    df.rename(columns=col_map, inplace=True)
    for col in ['open', 'close', 'high', 'low']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
# 核心技术指标计算
# ============================================================

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算移动平均线、振幅等技术指标"""
    df = df.copy()
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    df['amplitude'] = ((df['high'] - df['low']) / df['open'] * 100)
    if 'volume' in df.columns and len(df) >= 5:
        df['vol_ma5'] = df['volume'].rolling(5).mean()
    return df


def _detect_signals(df: pd.DataFrame) -> tuple:
    """检测买入/卖出信号，返回 (buy_list, sell_list)"""
    buy_signals = []
    sell_signals = []

    for i in range(2, len(df)):
        if pd.isna(df['MA5'].iloc[i]) or pd.isna(df['MA10'].iloc[i]):
            continue

        # 金叉：MA5 上穿 MA10
        golden_cross = (df['MA5'].iloc[i] > df['MA10'].iloc[i] and
                        df['MA5'].iloc[i - 1] <= df['MA10'].iloc[i - 1] and
                        df['close'].iloc[i] > df['close'].iloc[i - 1])
        # 放量金叉更可靠
        vol_surge = False
        if 'vol_ma5' in df.columns and not pd.isna(df['vol_ma5'].iloc[i]):
            vol_surge = df['volume'].iloc[i] > df['vol_ma5'].iloc[i] * 1.3

        if golden_cross:
            reason = 'MA5上穿MA10金叉' + (' + 放量' if vol_surge else '')
            buy_signals.append({'date': df['date'].iloc[i], 'price': df['close'].iloc[i], 'reason': reason})

        # 死叉：MA5 下穿 MA10
        death_cross = (df['MA5'].iloc[i] < df['MA10'].iloc[i] and
                       df['MA5'].iloc[i - 1] >= df['MA10'].iloc[i - 1])
        if death_cross:
            sell_signals.append(
                {'date': df['date'].iloc[i], 'price': df['close'].iloc[i], 'reason': 'MA5下穿MA10死叉'})

        # 长上影线顶部信号
        body_high = df[['open', 'close']].iloc[i].max()
        upper_shadow = df['high'].iloc[i] - body_high
        total_range = df['high'].iloc[i] - df['low'].iloc[i]
        if total_range > 0 and upper_shadow > total_range * 0.45 and df['close'].iloc[i] < df['open'].iloc[i]:
            sell_signals.append(
                {'date': df['date'].iloc[i], 'price': df['close'].iloc[i], 'reason': '长上影线顶部反转信号'})

        # 长下影线底部信号
        body_low = df[['open', 'close']].iloc[i].min()
        lower_shadow = body_low - df['low'].iloc[i]
        if total_range > 0 and lower_shadow > total_range * 0.45 and df['close'].iloc[i] > df['open'].iloc[i]:
            buy_signals.append(
                {'date': df['date'].iloc[i], 'price': df['close'].iloc[i], 'reason': '长下影线底部反转信号'})

    return buy_signals, sell_signals


# ============================================================
# 可视化工具 —— 核心：日波动+周波动 + 买卖建议
# ============================================================

@tool
def plot_stock_analysis(code: str, stock_name: str = "") -> str:
    """
    绘制股票日波动与周波动综合分析图，并基于波动大小和技术指标给出买入/卖出最佳时间建议。
    图表包含：日K线走势+波动区间+成交量、周K线走势+波动区间、移动平均线、买卖信号标注。

    code: 股票代码，如 '600519'
    stock_name: 股票名称（可选），如 '贵州茅台'
    返回: 图表保存路径 + 完整分析报告文本
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    day_raw = _get_day_kline(code, start_date, end_date, 1)
    week_raw = _get_week_kline(code, start_date, end_date, 1)

    day_records = _parse_kline_records(day_raw)
    week_records = _parse_kline_records(week_raw)

    if not day_records:
        return f"[错误] 未获取到 {code} 的日K线数据，请检查股票代码是否正确。"

    df_day = _normalize_kline_df(day_records)
    df_week = _normalize_kline_df(week_records) if week_records else pd.DataFrame()

    required = ['open', 'close', 'high', 'low']
    missing = [c for c in required if c not in df_day.columns]
    if missing:
        return f"[错误] 日K线数据缺少必要列: {missing}，实际列: {list(df_day.columns)}"

    # 计算指标
    df_day = _compute_indicators(df_day)
    if not df_week.empty:
        df_week = _compute_indicators(df_week)

    # 检测信号
    day_buy, day_sell = _detect_signals(df_day)

    # 近期信号（最近20个交易日）
    recent_cutoff = df_day['date'].max() - pd.Timedelta(days=20)
    recent_buys = [s for s in day_buy if s['date'] >= recent_cutoff]
    recent_sells = [s for s in day_sell if s['date'] >= recent_cutoff]

    # 最新价格与技术位
    latest = df_day.iloc[-1]
    latest_close = float(latest['close'])
    ma5_val = float(latest['MA5']) if not pd.isna(latest['MA5']) else latest_close
    ma10_val = float(latest['MA10']) if not pd.isna(latest['MA10']) else latest_close
    ma20_val = float(latest['MA20']) if not pd.isna(latest['MA20']) else latest_close

    # 趋势判断
    short_trend = "上升" if ma5_val > ma10_val else "下降"
    mid_trend = "上升" if ma10_val > ma20_val else "下降"

    # 波动统计
    recent_5d = df_day.tail(5)
    avg_amp_day = float(recent_5d['amplitude'].mean())
    if not df_week.empty and len(df_week) >= 4:
        avg_amp_week = float(df_week['amplitude'].tail(4).mean())
    else:
        avg_amp_week = avg_amp_day * 2.5

    # ---- 生成操作建议 ----
    title = f"{stock_name}({code})" if stock_name else f"股票{code}"

    if short_trend == "上升" and mid_trend == "上升":
        if recent_buys:
            recommendation = (
                f"【强烈买入】短期与中期趋势共振向上，近期出现买入信号。\n"
                f"  最佳买入时机：回调至 MA10({ma10_val:.2f}) 附近逢低吸纳\n"
                f"  止损位：MA20({ma20_val:.2f}) 下方 3%，即 {ma20_val * 0.97:.2f}"
            )
        else:
            recommendation = (
                f"【买入】趋势向好，暂无明确回调信号。\n"
                f"  建议沿 MA5({ma5_val:.2f}) 分批建仓\n"
                f"  止损位：{min(float(df_day['low'].tail(10).min()), ma20_val * 0.97):.2f}"
            )
    elif short_trend == "上升" and mid_trend == "下降":
        recommendation = (
            f"【谨慎观望/轻仓试多】短期反弹但中期仍处下行通道。\n"
            f"  激进者可轻仓试多，止损设在近期低点 {float(df_day['low'].tail(5).min()):.2f}\n"
            f"  稳健者等待 MA10 上穿 MA20 确认中期反转后再入场"
        )
    elif short_trend == "下降" and mid_trend == "上升":
        if recent_sells:
            recommendation = (
                f"【减仓/止盈】中期趋势向上但短期出现卖出信号。\n"
                f"  建议减仓 50%，观察 MA20({ma20_val:.2f}) 支撑力度\n"
                f"  若在 MA20 处企稳并出现金叉，可重新加仓"
            )
        else:
            recommendation = (
                f"【持有观望】中期上升趋势中的正常回调。\n"
                f"  关键支撑位：MA20({ma20_val:.2f})\n"
                f"  若放量跌破则止损，若企稳反弹则加仓"
            )
    else:
        recommendation = (
            f"【卖出/回避】短期与中期趋势均向下。\n"
            f"  建议离场观望，等待 MA5 上穿 MA10 金叉信号出现再考虑入场\n"
            f"  当前支撑参考：{float(df_day['low'].tail(20).min()):.2f}"
        )

    # ---- 波动大小分析 ----
    if avg_amp_day > 5:
        volatility_level = "高波动"
        vol_advice = "日振幅>5%，短线机会多但风险大，适合快进快出的波段操作"
    elif avg_amp_day > 2:
        volatility_level = "中等波动"
        vol_advice = "日振幅2%-5%，既有操作空间又不至于风险失控，适合趋势跟踪"
    else:
        volatility_level = "低波动"
        vol_advice = "日振幅<2%，波动较小，横盘整理概率大，适合观望或持有"

    # ---- 绘制图表 ----
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(20, 15))

    # ----- 图1: 日K线走势 + 波动区间 + MA -----
    ax1 = plt.subplot(4, 1, (1, 2))
    ax1.fill_between(df_day['date'], df_day['low'], df_day['high'],
                     alpha=0.25, color='#90CAF9', label='日波动区间 (最高-最低)')
    ax1.plot(df_day['date'], df_day['close'], color='#1565C0', linewidth=1.3, label='收盘价', zorder=3)
    ax1.plot(df_day['date'], df_day['MA5'], color='#FF6D00', linewidth=0.9, alpha=0.85, label='MA5')
    ax1.plot(df_day['date'], df_day['MA10'], color='#2E7D32', linewidth=0.9, alpha=0.85, label='MA10')
    ax1.plot(df_day['date'], df_day['MA20'], color='#C62828', linewidth=0.9, alpha=0.85, label='MA20')

    # 标注买卖信号
    for s in day_buy[-12:]:
        ax1.scatter(s['date'], s['price'], marker='^', color='#D50000', s=150,
                    zorder=6, edgecolors='white', linewidths=0.8)
        ax1.annotate('B', (s['date'], s['price']), textcoords="offset points",
                     xytext=(0, 14), fontsize=8, color='#D50000', ha='center', fontweight='bold')
    for s in day_sell[-12:]:
        ax1.scatter(s['date'], s['price'], marker='v', color='#00C853', s=150,
                    zorder=6, edgecolors='white', linewidths=0.8)
        ax1.annotate('S', (s['date'], s['price']), textcoords="offset points",
                     xytext=(0, -16), fontsize=8, color='#00695C', ha='center', fontweight='bold')

    ax1.set_ylabel('价格 (元)', fontsize=11)
    ax1.set_title(f'{title}  ┃  日波动 & 买卖信号', fontsize=15, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8, ncol=6)
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # ----- 图2: 日成交量 -----
    ax_vol = plt.subplot(4, 1, 3)
    if 'volume' in df_day.columns:
        colors = ['#EF5350' if df_day['close'].iloc[i] >= df_day['open'].iloc[i]
                  else '#26A69A' for i in range(len(df_day))]
        ax_vol.bar(df_day['date'], df_day['volume'], color=colors, width=0.8, alpha=0.75)
        if 'vol_ma5' in df_day.columns:
            ax_vol.plot(df_day['date'], df_day['vol_ma5'], color='#FF6D00', linewidth=1, alpha=0.7, label='量MA5')
        ax_vol.set_ylabel('成交量', fontsize=10)
        ax_vol.set_title('日成交量 (红涨绿跌)', fontsize=12, fontweight='bold')
        ax_vol.legend(loc='upper left', fontsize=8)
        ax_vol.grid(True, alpha=0.25)
        ax_vol.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # ----- 图3: 周K线走势 + 周波动区间 -----
    ax_w = plt.subplot(4, 1, 4)
    if not df_week.empty:
        ax_w.fill_between(df_week['date'], df_week['low'], df_week['high'],
                          alpha=0.3, color='#B39DDB', label='周波动区间')
        ax_w.plot(df_week['date'], df_week['close'], color='#4527A0', linewidth=1.5,
                  marker='o', markersize=5, label='周收盘价')
        for ma_name, ma_col, ma_ls in [('MA5', '#FF6D00', '--'), ('MA10', '#2E7D32', '--'), ('MA20', '#C62828', '--')]:
            if ma_name in df_week.columns:
                ax_w.plot(df_week['date'], df_week[ma_name], color=ma_col,
                          linewidth=0.9, linestyle=ma_ls, alpha=0.8, label=f'周{ma_name}')

        # 标注周级别最高/最低
        wh = df_week['high'].idxmax()
        wl = df_week['low'].idxmin()
        if pd.notna(wh):
            ax_w.annotate(f"周高 {df_week['high'].max():.2f}",
                          (df_week['date'].loc[wh], df_week['high'].max()),
                          fontsize=7.5, color='#C62828', ha='center', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        if pd.notna(wl):
            ax_w.annotate(f"周低 {df_week['low'].min():.2f}",
                          (df_week['date'].loc[wl], df_week['low'].min()),
                          fontsize=7.5, color='#2E7D32', ha='center', va='top',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        ax_w.set_ylabel('价格 (元)', fontsize=10)
        ax_w.set_title('周波动走势分析', fontsize=13, fontweight='bold')
        ax_w.legend(loc='upper left', fontsize=8, ncol=6)
        ax_w.grid(True, alpha=0.25)
        ax_w.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    plt.tight_layout(pad=2)

    # 保存图表
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"stock_{code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ---- 构建分析报告 ----
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║              {title} 综合分析报告
╚══════════════════════════════════════════════════════════════╝

  【价格速览】
    最新收盘价 : {latest_close:.2f} 元
    MA5  (5日均线)  : {ma5_val:.2f}
    MA10 (10日均线) : {ma10_val:.2f}
    MA20 (20日均线) : {ma20_val:.2f}

  【波动分析】
    近5日 平均日振幅 : {avg_amp_day:.2f}%
    近4周 平均周振幅 : {avg_amp_week:.2f}%
    波动评级 : {volatility_level}
    → {vol_advice}

  【趋势判断】
    短期 (MA5↔MA10) : {short_trend}
    中期 (MA10↔MA20): {mid_trend}

  【近期买卖信号】
    买入信号 ({len(recent_buys)} 个):
{chr(10).join(f'      ▸ {s["date"].strftime("%m-%d")}  {s["reason"]}  @{s["price"]:.2f}' for s in recent_buys[:5]) if recent_buys else '      暂无买入信号'}
    卖出信号 ({len(recent_sells)} 个):
{chr(10).join(f'      ▸ {s["date"].strftime("%m-%d")}  {s["reason"]}  @{s["price"]:.2f}' for s in recent_sells[:5]) if recent_sells else '      暂无卖出信号'}

  【最佳操作建议】
  ┌─────────────────────────────────────────────────────────┐
  │ {recommendation.replace(chr(10), chr(10) + '  │ ')}│
  └─────────────────────────────────────────────────────────┘

  免责声明：以上分析基于技术指标自动计算，仅供参考。股市有风险，投资需谨慎。

  图表已保存至: {filepath}
"""
    return report


# ============================================================
# 构建 DeepAgent
# ============================================================

def build_agent():
    """构建股票分析 DeepAgent"""

    tools = [
        search_stock,
        get_stock_info,
        get_day_kline,
        get_week_kline,
        get_month_kline,
        get_stock_rank,
        get_board_info,
        get_stock_minute_data,
        plot_stock_analysis,
    ]

    system_prompt = """你是一个专业的股票技术分析助手。你有以下工具可用：

【数据查询工具】
- search_stock: 通过名称或代码模糊搜索股票
- get_stock_info: 获取股票基础信息（行业、市值等）
- get_day_kline / get_week_kline / get_month_kline: 获取日/周/月K线
- get_stock_rank: 查看股票涨幅排行
- get_board_info: 查看大盘指数
- get_stock_minute_data: 获取分时数据

【综合分析工具（核心）】
- plot_stock_analysis: 一键生成包含日波动+周波动+成交量+买卖信号的可视化分析图，并给出操作建议

## 工作流程

当用户询问某只股票时：
1. 若用户给的是股票名称而非代码 → 先用 search_stock 查代码
2. 使用 plot_stock_analysis 生成综合分析图和报告（此工具会自动拉取日K和周K数据）
3. 根据需要补充月K线分析判断长期趋势

## 重要
- 始终基于实际数据分析，不要凭空猜测价格
- 每次回答末尾提醒：分析仅供参考，股市有风险，投资需谨慎
- 使用中文回复，格式清晰易读
"""

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        backend=FilesystemBackend(
            root_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_state")
        ),
    )
    return agent


# ============================================================
# 命令行入口
# ============================================================

def _run_interactive(agent):
    """交互式对话循环"""
    console.print("[bold]交互模式已启动（输入 'exit' 退出，输入 'help' 查看示例）[/bold]\n")
    console.print("[dim]示例查询：[/dim]")
    console.print("  • 帮我分析一下贵州茅台")
    console.print("  • 搜索股票 比亚迪 并分析")
    console.print("  • 查看今天的股票涨幅排行")
    console.print("  • 帮我看看大盘情况\n")

    while True:
        try:
            user_input = input("请输入查询: ").strip()
            if user_input.lower() in ('exit', 'quit', 'q'):
                console.print("[yellow] 再见！[/yellow]")
                break
            if not user_input:
                continue
            if user_input.lower() == 'help':
                console.print("[dim]示例：分析贵州茅台 / 搜索宁德时代 / 查看涨幅排行 / 大盘怎么样[/dim]")
                continue

            with console.status("[cyan]分析中，请稍候...[/cyan]"):
                result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})

            if isinstance(result, dict) and "messages" in result:
                last_msg = result["messages"][-1]
                _print_result(last_msg.content)
            else:
                console.print(result)

        except KeyboardInterrupt:
            console.print("\n[yellow] 再见！[/yellow]")
            break
        except Exception as e:
            console.print(f"[red][错误] 出错: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="股票信息查询与可视化分析 Agent")
    parser.add_argument("-q", "--query", type=str, help="直接查询，如 '分析贵州茅台'")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]股票分析 DeepAgent[/bold cyan]\n"
        "[dim]基于 LangChain DeepAgent 框架 | 数据: api.autostock.cn[/dim]\n"
        "[dim]功能: 股票搜索 · K线数据 · 日/周波动可视化 · 买卖信号检测 · 操作建议[/dim]",
        border_style="cyan"
    ))

    agent = build_agent()
    console.print("[green][OK] Agent 就绪[/green]\n")

    if args.query:
        console.print(f"[yellow]{args.query}[/yellow]\n")
        with console.status("[cyan]分析中...[/cyan]"):
            result = agent.invoke({"messages": [{"role": "user", "content": args.query}]})
        if isinstance(result, dict) and "messages" in result:
            last_msg = result["messages"][-1]
            _print_result(last_msg.content)
        else:
            console.print(result)
    else:
        _run_interactive(agent)


if __name__ == "__main__":
    main()
