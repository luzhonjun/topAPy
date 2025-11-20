import os
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any
import subprocess

from flask import Flask, request, jsonify
from flask_cors import CORS

# 延迟导入，避免无环境时报错
import importlib
import time


app = Flask(__name__)
CORS(app)


def _load_tushare():
    return importlib.import_module('tushare')


 


@lru_cache(maxsize=1)
def _stock_name_map() -> Dict[str, str]:
    ts = _load_tushare()
    token = os.getenv('TUSHARE_TOKEN')
    pro = ts.pro_api(token) if token else ts.pro_api()
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
    return {row['ts_code']: row['name'] for _, row in df.iterrows()}


def _last_trade_date() -> str:
    ts = _load_tushare()
    token = os.getenv('TUSHARE_TOKEN')
    pro = ts.pro_api(token) if token else ts.pro_api()
    today = datetime.now().strftime('%Y%m%d')
    cal = pro.trade_cal(exchange='SSE', start_date='20000101', end_date=today, is_open=1)
    if cal.empty:
        # 兜底：使用今天日期
        return datetime.now().strftime('%Y-%m-%d')
    cal = cal.sort_values(by='cal_date')
    last_open = cal.iloc[-1]['cal_date']
    return f"{last_open[:4]}-{last_open[4:6]}-{last_open[6:]}"


_daily_cache: Dict[str, Dict[str, Any]] = {}


@app.route('/api/stocks/top/daily')
def get_top_daily():
    try:
        limit = int(request.args.get('limit', '50'))
        date_arg = request.args.get('date')
        offset_arg = int(request.args.get('offset', '0'))
        ts = _load_tushare()
        token = os.getenv('TUSHARE_TOKEN')
        pro = ts.pro_api(token) if token else ts.pro_api()

        if date_arg:
            trade_date = date_arg
        else:
            today = datetime.now().strftime('%Y%m%d')
            cal = pro.trade_cal(exchange='SSE', start_date='20000101', end_date=today, is_open=1)
            cal = cal.sort_values(by='cal_date')
            open_dates = cal['cal_date'].tolist()
            if not open_dates:
                trade_date = datetime.now().strftime('%Y-%m-%d')
            else:
                idx = max(0, min(len(open_dates) - 1 + offset_arg, len(open_dates) - 1))
                d = open_dates[idx]
                trade_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        trade_date_compact = trade_date.replace('-', '')

        cache_key = trade_date_compact
        now_ts = time.time()
        cache_item = _daily_cache.get(cache_key)
        if cache_item and now_ts - cache_item.get('ts', 0) < 3600:
            return jsonify({'date': cache_item['date'], 'list': cache_item['list'][:limit]})

        try:
            df = pro.daily(trade_date=trade_date_compact)
            if df.empty:
                if not date_arg and offset_arg < 0:
                    idx = open_dates.index(trade_date_compact) if trade_date_compact in open_dates else len(open_dates) - 1
                    prev_idx = max(0, idx - 1)
                    d = open_dates[prev_idx]
                    fallback_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                    df = pro.daily(trade_date=d)
                    trade_date = fallback_date
                    trade_date_compact = d
                if df.empty:
                    return jsonify({'date': trade_date, 'list': [], 'message': 'no data for trade_date'})

            if 'ts_code' in df.columns:
                df = df[df['ts_code'].str.endswith('.SH') | df['ts_code'].str.endswith('.SZ')].copy()
                df['__code'] = df['ts_code'].str.split('.').str[0]
                df = df[~df['__code'].str.startswith('200') & ~df['__code'].str.startswith('900')]
            name_map = _stock_name_map()

            records: List[Dict[str, Any]] = []
            for _, row in df.iterrows():
                ts_code = row['ts_code']
                code = ts_code.split('.')[0]
                name = name_map.get(ts_code, '')
                amount_yuan = float(row['amount']) * 1000.0

                records.append({
                    'code': code,
                    'name': name,
                    'price': float(row['close']),
                    'change': float(row['change']),
                    'changePercent': float(row['pct_chg']),
                    'volume': int(float(row['vol']) * 100.0),
                    'amount': int(amount_yuan),
                    'date': trade_date
                })

            records.sort(key=lambda x: x['amount'], reverse=True)
            top_list = records[:limit]

            resp = {'date': trade_date, 'list': top_list}
            _daily_cache[cache_key] = {'date': trade_date, 'list': records, 'ts': now_ts}
            return jsonify(resp)
        except Exception as e:
            msg = str(e)
            cache_item = _daily_cache.get(cache_key)
            if cache_item:
                return jsonify({'date': cache_item['date'], 'list': cache_item['list'][:limit], 'message': 'served from cache'})
            try:
                if not date_arg and offset_arg < 0:
                    idx = open_dates.index(trade_date_compact) if trade_date_compact in open_dates else len(open_dates) - 1
                    prev_idx = max(0, idx - 1)
                    d = open_dates[prev_idx]
                    fallback_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                    df_fb = pro.daily(trade_date=d)
                    if not df_fb.empty:
                        name_map = _stock_name_map()
                        records: List[Dict[str, Any]] = []
                        for _, row in df_fb.iterrows():
                            ts_code = row['ts_code']
                            code = ts_code.split('.')[0]
                            name = name_map.get(ts_code, '')
                            amount_yuan = float(row['amount']) * 1000.0
                            records.append({
                                'code': code,
                                'name': name,
                                'price': float(row['close']),
                                'change': float(row['change']),
                                'changePercent': float(row['pct_chg']),
                                'volume': int(float(row['vol']) * 100.0),
                                'amount': int(amount_yuan),
                                'date': fallback_date
                            })
                        records.sort(key=lambda x: x['amount'], reverse=True)
                        _daily_cache[d] = {'date': fallback_date, 'list': records, 'ts': now_ts}
                        return jsonify({'date': fallback_date, 'list': records[:limit], 'message': 'fallback to previous trading day'})
            except Exception:
                pass
            return jsonify({'date': trade_date, 'list': [], 'message': msg})
    except Exception as e:
        return jsonify({'date': '', 'list': [], 'message': str(e)})




if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=False)