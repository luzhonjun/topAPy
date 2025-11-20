# A股成交额排行榜

一个基于Vue3的A股成交额排行榜单页面应用，支持查看今日和昨日的前50/100名股票，并可导出Excel数据。

## 功能特性

- 📊 展示A股成交额前50/100名股票
- 📅 支持切换今日和昨日数据
- 📈 显示股票价格、涨跌幅、成交量等关键信息
- 💾 支持导出Excel文件
- 📱 响应式设计，支持移动端
- 🎨 现代化UI界面

## 技术栈

- Vue 3 + TypeScript
- Vite
- Tailwind CSS
- XLSX (Excel导出)
- Axios (HTTP请求)

## 快速开始

### 安装依赖
```bash
npm install
```

### 启动开发服务器
```bash
npm run dev
```

### 构建生产版本
```bash
npm run build
```

## 使用说明

1. **选择数据时间**：可以通过单选按钮切换"今日"或"昨日"数据
2. **选择排行榜范围**：可以选择显示前50名或前100名股票
3. **查看股票信息**：表格显示股票代码、名称、价格、涨跌幅、成交量、成交额等信息
4. **导出Excel**：点击"导出Excel"按钮将当前数据导出为Excel文件

## 集成真实Baostock API

当前版本使用模拟数据，如需集成真实的Baostock API，请按以下步骤操作：

### 方法1：创建Python后端服务

由于Baostock是Python库，建议创建一个Python后端服务：

1. 安装Python依赖：
```bash
pip install baostock flask flask-cors
```

2. 创建Python API服务 (`stock_api.py`)：
```python
import baostock as bs
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

def get_top_stocks_by_amount(limit=50, date=None):
    """获取成交额前N的股票"""
    # 登录Baostock
    lg = bs.login()
    if lg.error_code != '0':
        return []
    
    try:
        # 获取A股所有股票
        rs = bs.query_all_stock(day=date or datetime.now().strftime('%Y-%m-%d'))
        
        stocks = []
        while rs.error_code == '0' and rs.next():
            stock_code = rs.get_row_data()[0]
            if stock_code.startswith('sh') or stock_code.startswith('sz'):
                # 获取股票行情数据
                quote_rs = bs.query_history_k_data_plus(
                    stock_code,
                    "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
                    start_date=date or datetime.now().strftime('%Y-%m-%d'),
                    end_date=date or datetime.now().strftime('%Y-%m-%d'),
                    frequency="d",
                    adjustflag="3"
                )
                
                if quote_rs.error_code == '0':
                    data = quote_rs.get_row_data()
                    if data and float(data[8]) > 0:  # 成交额大于0
                        stocks.append({
                            'code': stock_code[3:],  # 去掉sh/sz前缀
                            'name': '',  # 需要另外获取股票名称
                            'price': float(data[5]),
                            'change': float(data[5]) - float(data[6]),
                            'changePercent': float(data[10]) if data[10] else 0,
                            'volume': int(data[7]),
                            'amount': float(data[8]),
                            'date': data[0]
                        })
        
        # 按成交额排序并返回前N名
        stocks.sort(key=lambda x: x['amount'], reverse=True)
        return stocks[:limit]
        
    finally:
        bs.logout()

@app.route('/api/stocks/top/<int:limit>')
def get_today_top_stocks(limit):
    """获取今日成交额前N的股票"""
    stocks = get_top_stocks_by_amount(limit)
    return jsonify(stocks)

@app.route('/api/stocks/yesterday/top/<int:limit>')
def get_yesterday_top_stocks(limit):
    """获取昨日成交额前N的股票"""
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    stocks = get_top_stocks_by_amount(limit, yesterday)
    return jsonify(stocks)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

3. 修改前端服务 (`src/services/stockService.ts`)：
```typescript
// 替换getTopStocksByAmount方法
static async getTopStocksByAmount(limit: number = 50): Promise<StockData[]> {
  try {
    const response = await axios.get(`http://localhost:5000/api/stocks/top/${limit}`)
    return response.data
  } catch (error) {
    console.error('获取股票数据失败:', error)
    throw error
  }
}

// 替换getYesterdayTopStocksByAmount方法
static async getYesterdayTopStocksByAmount(limit: number = 50): Promise<StockData[]> {
  try {
    const response = await axios.get(`http://localhost:5000/api/stocks/yesterday/top/${limit}`)
    return response.data
  } catch (error) {
    console.error('获取昨日股票数据失败:', error)
    throw error
  }
}
```

### 方法2：使用第三方股票API

如果不想搭建Python服务，也可以使用其他第三方股票API，如：
- 新浪财经API
- 腾讯股票API
- 网易股票API
- 聚合数据股票API

## 注意事项

1. **Baostock限制**：Baostock有访问频率限制，请合理控制请求频率
2. **数据延迟**：股票数据可能有15分钟延迟
3. **交易时间**：A股交易时间为工作日9:30-11:30，13:00-15:00
4. **网络代理**：如果在国外访问，可能需要设置网络代理

## 项目结构

```
src/
├── components/
│   └── StockList.vue          # 主要股票列表组件
├── services/
│   ├── stockService.ts        # 股票数据服务
│   └── excelExportService.ts  # Excel导出服务
├── App.vue                    # 主应用组件
└── main.ts                    # 应用入口
```

## 许可证

MIT License
