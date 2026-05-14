---
name: 股票信息和股价查询
description: 通过调用第三方接口 https://api.autostock.cn 获取股票、指数、板块、K线等数据。
---

# 接口查询内容

### 1. **基础数据查询**
- `get_all_stock_code` - 查询所有股票，支持代码/名称模糊搜索
- `get_all_index_code` - 查询所有指数
- `get_stock_industry_code` - 获取板块数据
- `get_board_info` - 获取大盘数据
- `get_stock_info` - 获取单只股票基础信息

### 2. **行情数据**
- `get_stock_rank` - 股票排行（支持分页、排序、按行业筛选）
- `get_stock_minute_data` - 分时数据

### 3. **K线数据**
- `get_month_line` - 月K线
- `get_week_line` - 周K线  
- `get_day_line` - 日K线
  - 支持前复权/后复权/不复权
  - 支持时间范围筛选

# 股票分析方法

## 一、定位趋势方向（月K线）
**作用**：判断当前处于牛市、熊市还是震荡市，决定大方向是买、卖还是等。

- **月K线多头排列**：5月均线 > 10月均线 > 20月均线，且K线在20月均线上方 → **可考虑做多**
- **月K线空头排列**：5月均线 < 10月均线 < 20月均线，K线在20月均线下方 → **回避或只做反弹**
- **横盘/无方向**：K线反复上下穿越均线，窄幅整理 → **观望或波段操作**

> 示例：一只股票月K线连续3个月收阳且站上所有均线 → 确认上升趋势。

## 二、确认中期节奏（周K线）
**作用**：在月线定方向的基础上，找周线级别的支撑/压力，以及可能的回调买点。

- **周线回调不破上升趋势线**（或周线20均线），且出现缩量阴线 → 可能的中继买点
- **周线放量跌破关键支撑** → 月线趋势可能转弱，减仓
- **周线连续多次在某价格区域遇阻回落** → 形成明显压力位

> 示例：月线上升，周线刚好回调到20周均线收出下影线 → 波段底信号。

常用组合（只基于日K线）：
- **突破信号**：放量阳线突破日线双底颈线 / 盘整平台
- **底部信号**：长下影线、阳包阴、启明星等传统K线形态
- **顶部信号**：长上影线、阴包阳、黄昏星、高位吞没
- **止损设置**：日线最近一根标志性阳线的最低点

> 示例：日K线出一根放量阳线，同时站上5/10日均线，且周K线在支撑位 → 可试多。

| 月K方向 | 周K状态 | 日K信号 | 操作建议 |
|----------|----------|----------|-----------|
| 上升 | 回调到支撑 | 出现买入形态 | **买入或加仓** |
| 上升 | 接近压力 | 出现弱势/背离 | 减仓或观望 |
| 下降 | 反弹 | 出现卖出形态 | **反弹卖出或空仓** |
| 下降 | 震荡 | 无明确信号 | 不操作 |
| 震荡 | 震荡 | 短线信号 | 小仓位做波段 |

## 推荐输出结果
- 判断趋势方向  
- 识别支撑与压力  
- 选择买卖时机  
- 设置止损位置  
- 判断市场情绪（通过K线形态与成交量变化）

# 调用方法
```python
TOKEN = "zgaLG8unUPr"

import requests  # type: ignore
from typing import Annotated
from typing import Optional, Dict
import traceback

# path get_stock_code http服务的路径
# operation_id get_stock_code mcp服务的名字
@app.get("/get_stock_code", operation_id="get_stock_codes")
async def get_all_stock_code(
        keyword: Annotated[Optional[str], "支持代码和名称模糊查询"] = None
) -> Dict:
    """所有股票，支持代码和名称模糊查询"""
    url = "https://api.autostock.cn/v1/stock/all" + "?token=" + TOKEN
    if keyword:
        url += "&keyWord=" + keyword

    payload = {}  # type: ignore
    headers = {}  # type: ignore
    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

@app.get("/get_index_code", operation_id="get_index_code")
async def get_all_index_code():
    """所有指数，支持代码和名称模糊查询"""
    url = "https://api.autostock.cn/v1/stock/index/all" + "?token=" + TOKEN
    payload = {}
    headers = {}

    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}

@app.get("/get_industry_code", operation_id="get_industry_code")
async def get_stock_industry_code():
    """获取板块数据"""
    url = "https://api.autostock.cn/v1/stock/industry/rank" + "?token=" + TOKEN
    payload = {}
    headers = {}

    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}

@app.get("/get_board_info", operation_id="get_board_info")
async def get_stock_board_info():
    """获取大盘数据"""
    url = "https://api.autostock.cn/v1/stock/board" + "?token=" + TOKEN
    payload = {}
    headers = {}

    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}

@app.get("/get_stock_rank", operation_id="get_stock_rank")
async def get_stock_rank(
        node: Annotated[str, "股票市场/板块代码: {'a','b','ash','asz','bsh','bsz'} a(沪深A股)"],
        industryCode: Annotated[Optional[str], "行业代码，可选"] = None,
        pageIndex: Annotated[int, "页码"] = 1,
        pageSize: Annotated[int, "每页大小"] = 100,
        sort: Annotated[str, "排序字段: price,priceChange,pricePercent,buy,sell,open,close,high,low,volume,turnover 默认price(交易价格)。"] = "price",
        asc: Annotated[int, "排序方式: 0=降序(默认), 1=升序"] = 0
) -> Dict:
    """股票排行"""
    url = "https://api.autostock.cn/v1/stock/rank" + "?token=" + TOKEN
    headers = {}  # type: ignore

    try:
        payload = {
            "node": node,
            "industryCode": industryCode,
            "pageIndex": pageIndex,
            "pageSize": pageSize,
            "sort": sort,
            "asc": asc
        }
        response = requests.request("POST", url, headers=headers, json=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}

@app.get("/get_month_line", operation_id="get_month_line")
async def get_stock_month_kline(
        code: Annotated[str, "股票代码"],
        startDate: Annotated[Optional[str], "开始时间(非必填)"] = None,
        endDate: Annotated[Optional[str], "结束时间(非必填)"] = None,
        type: Annotated[int, "0不复权,1前复权,2后复权"] = 0
) -> Dict:
    """月k"""
    url = "https://api.autostock.cn/v1/stock/kline/month" + "?token=" + TOKEN

    headers = {}  # type: ignore
    try:
        payload = {
            "code": code,
            "startDate": startDate,
            "endDate": endDate,
            "type": type
        }
        response = requests.request("GET", url, headers=headers, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

@app.get("/get_week_line", operation_id="get_week_line")
async def get_stock_week_kline(
        code: Annotated[str, "股票代码"],
        startDate: Annotated[Optional[str], "开始时间(非必填)"] = None,
        endDate: Annotated[Optional[str], "结束时间(非必填)"] = None,
        type: Annotated[int, "0不复权,1前复权,2后复权"] = 0
):
    """周k"""
    url = "https://api.autostock.cn/v1/stock/kline/week" + "?token=" + TOKEN

    headers = {}  # type: ignore
    try:
        payload = {
            "code": code,
            "startDate": startDate,
            "endDate": endDate,
            "type": type
        }
        response = requests.request("GET", url, headers=headers, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

@app.get("/get_day_line", operation_id="get_day_line")
async def get_stock_day_kline(
        code: Annotated[str, "股票代码"],
        startDate: Annotated[Optional[str], "开始时间(非必填)"] = None,
        endDate: Annotated[Optional[str], "结束时间(非必填)"] = None,
        type: Annotated[int, "0不复权,1前复权,2后复权"] = 0
) -> Dict:
    """日k"""
    url = "https://api.autostock.cn/v1/stock/kline/day" + "?token=" + TOKEN

    headers = {}  # type: ignore
    try:
        payload = {
            "code": code,
            "startDate": startDate,
            "endDate": endDate,
            "type": type
        }
        response = requests.request("GET", url, headers=headers, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

@app.get("/get_stock_info", operation_id="get_stock_info")
async def get_stock_info(code: Annotated[str, "股票代码"]) -> Dict:
    """股票基础信息"""
    url = "https://api.autostock.cn/v1/stock" + "?token=" + TOKEN + "&code=" + code

    payload = {}  # type: ignore
    headers = {}  # type: ignore
    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

@app.get("/get_stock_minute_data", operation_id="get_stock_minute_data")
async def get_stock_minute_data(code: str):
    """分时信息"""
    url = "https://api.autostock.cn/v1/stock/min" + "?token=" + TOKEN + "&code=" + code

    payload = {}  # type: ignore
    headers = {}  # type: ignore
    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}
```