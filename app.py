import os
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS

# 延迟导入，避免无环境时报错
import importlib
import time
import psycopg
from psycopg.rows import dict_row


app = Flask(__name__)
CORS(app)
app.url_map.strict_slashes = False

@app.route('/healthz')
def healthz():
    return jsonify({'status': 'ok'})


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

def _get_db_conn():
    dsn = os.getenv('DATABASE_URL')
    if not dsn:
        return None
    try:
        return psycopg.connect(dsn, autocommit=True)
    except Exception:
        return None

def _ensure_db_schema():
    conn = _get_db_conn()
    if not conn:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_rankings (
              id SERIAL PRIMARY KEY,
              trade_date date NOT NULL,
              code text NOT NULL,
              name text,
              price numeric,
              change numeric,
              change_percent numeric,
              volume bigint,
              amount bigint,
              rank int,
              created_at timestamptz DEFAULT now(),
              UNIQUE (trade_date, code)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trading_calendar (
              cal_date date NOT NULL,
              exchange text NOT NULL,
              is_open boolean NOT NULL,
              updated_at timestamptz DEFAULT now(),
              PRIMARY KEY (cal_date, exchange)
            );
            """
        )

try:
    _ensure_db_schema()
except Exception:
    pass

def _date_to_compact(d: str) -> str:
    return d.replace('-', '')

def _compact_to_date(s: str) -> str:
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"

def _upsert_rankings(trade_date: str, records: List[Dict[str, Any]]):
    conn = _get_db_conn()
    if not conn:
        return
    with conn.cursor() as cur:
        for idx, r in enumerate(records):
            cur.execute(
                """
                INSERT INTO stock_rankings (trade_date, code, name, price, change, change_percent, volume, amount, rank)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trade_date, code)
                DO UPDATE SET
                  name = EXCLUDED.name,
                  price = EXCLUDED.price,
                  change = EXCLUDED.change,
                  change_percent = EXCLUDED.change_percent,
                  volume = EXCLUDED.volume,
                  amount = EXCLUDED.amount,
                  rank = EXCLUDED.rank
                """,
                (
                    trade_date,
                    r['code'],
                    r['name'],
                    r['price'],
                    r['change'],
                    r['changePercent'],
                    r['volume'],
                    r['amount'],
                    idx + 1,
                ),
            )

def _get_top_from_db(date_str: str, limit: int) -> List[Dict[str, Any]]:
    conn = _get_db_conn()
    if not conn:
        return []
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT code, name, price, change, change_percent AS "changePercent",
                   volume, amount, trade_date AS "date", rank
            FROM stock_rankings
            WHERE trade_date = %s
            ORDER BY rank ASC
            LIMIT %s
            """,
            (date_str, limit),
        )
        rows = cur.fetchall()
    return rows

def _get_open_dates_from_db(end_compact: str, k: int) -> List[str]:
    conn = _get_db_conn()
    if not conn:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT to_char(cal_date, 'YYYYMMDD')
            FROM trading_calendar
            WHERE exchange = %s AND is_open = true AND cal_date <= to_date(%s, 'YYYYMMDD')
            ORDER BY cal_date DESC
            LIMIT %s
            """,
            ('SSE', end_compact, k),
        )
        res = cur.fetchall()
    dates = [r[0] for r in res]
    return list(reversed(dates))

def _sync_calendar(pro, start_compact: str, end_compact: str):
    conn = _get_db_conn()
    if not conn:
        return
    cal = pro.trade_cal(exchange='SSE', start_date=start_compact, end_date=end_compact, is_open=1)
    if cal.empty:
        return
    with conn.cursor() as cur:
        for _, row in cal.iterrows():
            cal_date = row['cal_date']
            cur.execute(
                """
                INSERT INTO trading_calendar (cal_date, exchange, is_open)
                VALUES (to_date(%s, 'YYYYMMDD'), %s, true)
                ON CONFLICT (cal_date, exchange)
                DO UPDATE SET is_open = EXCLUDED.is_open, updated_at = now()
                """,
                (cal_date, 'SSE'),
            )

def _build_daily_records_from_df(df, name_map, trade_date: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if 'ts_code' in df.columns:
        df = df[df['ts_code'].str.endswith('.SH') | df['ts_code'].str.endswith('.SZ')].copy()
        df['__code'] = df['ts_code'].str.split('.').str[0]
        df = df[~df['__code'].str.startswith('200') & ~df['__code'].str.startswith('900')]
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
    return records

def _get_open_dates(pro, end_compact: str, k: int) -> List[str]:
    cal = pro.trade_cal(exchange='SSE', start_date='20000101', end_date=end_compact, is_open=1)
    cal = cal.sort_values(by='cal_date')
    open_dates = cal['cal_date'].tolist()
    if not open_dates:
        return []
    return open_dates[-k:]


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
            _upsert_rankings(trade_date, records)
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




@app.route('/api/stocks/top/new')
def get_top_new():
    try:
        limit = int(request.args.get('limit', '50'))
        date_arg = request.args.get('date')
        ts = _load_tushare()
        token = os.getenv('TUSHARE_TOKEN')
        pro = ts.pro_api(token) if token else ts.pro_api()

        if date_arg:
            curr_date = date_arg
            curr_compact = date_arg.replace('-', '')
        else:
            today = datetime.now().strftime('%Y%m%d')
            cal = pro.trade_cal(exchange='SSE', start_date='20000101', end_date=today, is_open=1)
            cal = cal.sort_values(by='cal_date')
            open_dates = cal['cal_date'].tolist()
            if not open_dates:
                curr_date = datetime.now().strftime('%Y-%m-%d')
                curr_compact = datetime.now().strftime('%Y%m%d')
            else:
                d = open_dates[-1]
                curr_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                curr_compact = d

        cal_prev = pro.trade_cal(exchange='SSE', start_date='20000101', end_date=curr_compact, is_open=1)
        cal_prev = cal_prev.sort_values(by='cal_date')
        open_dates_prev = cal_prev['cal_date'].tolist()
        if not open_dates_prev:
            prev_compact = curr_compact
            prev_date = curr_date
        else:
            if curr_compact in open_dates_prev:
                idx = open_dates_prev.index(curr_compact)
                prev_idx = max(0, idx - 1)
            else:
                prev_idx = len(open_dates_prev) - 1
            dprev = open_dates_prev[prev_idx]
            prev_compact = dprev
            prev_date = f"{dprev[:4]}-{dprev[4:6]}-{dprev[6:]}"

        now_ts = time.time()
        name_map = _stock_name_map()

        cache_item_today = _daily_cache.get(curr_compact)
        if cache_item_today and now_ts - cache_item_today.get('ts', 0) < 3600:
            records_today = cache_item_today['list']
            curr_date_final = cache_item_today['date']
        else:
            df_today = pro.daily(trade_date=curr_compact)
            if df_today.empty:
                return jsonify({'date': curr_date, 'prev_date': prev_date, 'limit': limit, 'new': [], 'message': '该日期暂无数据'})
            if 'ts_code' in df_today.columns:
                df_today = df_today[df_today['ts_code'].str.endswith('.SH') | df_today['ts_code'].str.endswith('.SZ')].copy()
                df_today['__code'] = df_today['ts_code'].str.split('.').str[0]
                df_today = df_today[~df_today['__code'].str.startswith('200') & ~df_today['__code'].str.startswith('900')]
            records_today: List[Dict[str, Any]] = []
            for _, row in df_today.iterrows():
                ts_code = row['ts_code']
                code = ts_code.split('.')[0]
                name = name_map.get(ts_code, '')
                amount_yuan = float(row['amount']) * 1000.0
                records_today.append({
                    'code': code,
                    'name': name,
                    'price': float(row['close']),
                    'change': float(row['change']),
                    'changePercent': float(row['pct_chg']),
                    'volume': int(float(row['vol']) * 100.0),
                    'amount': int(amount_yuan),
                    'date': curr_date
                })
            records_today.sort(key=lambda x: x['amount'], reverse=True)
            _daily_cache[curr_compact] = {'date': curr_date, 'list': records_today, 'ts': now_ts}
            curr_date_final = curr_date

        cache_item_prev = _daily_cache.get(prev_compact)
        if cache_item_prev and now_ts - cache_item_prev.get('ts', 0) < 3600:
            records_prev = cache_item_prev['list']
            prev_date_final = cache_item_prev['date']
        else:
            df_prev = pro.daily(trade_date=prev_compact)
            if df_prev.empty:
                prev_date_final = prev_date
                records_prev = []
            else:
                if 'ts_code' in df_prev.columns:
                    df_prev = df_prev[df_prev['ts_code'].str.endswith('.SH') | df_prev['ts_code'].str.endswith('.SZ')].copy()
                    df_prev['__code'] = df_prev['ts_code'].str.split('.').str[0]
                    df_prev = df_prev[~df_prev['__code'].str.startswith('200') & ~df_prev['__code'].str.startswith('900')]
                records_prev: List[Dict[str, Any]] = []
                for _, row in df_prev.iterrows():
                    ts_code = row['ts_code']
                    code = ts_code.split('.')[0]
                    name = name_map.get(ts_code, '')
                    amount_yuan = float(row['amount']) * 1000.0
                    records_prev.append({
                        'code': code,
                        'name': name,
                        'price': float(row['close']),
                        'change': float(row['change']),
                        'changePercent': float(row['pct_chg']),
                        'volume': int(float(row['vol']) * 100.0),
                        'amount': int(amount_yuan),
                        'date': prev_date
                    })
                records_prev.sort(key=lambda x: x['amount'], reverse=True)
                _daily_cache[prev_compact] = {'date': prev_date, 'list': records_prev, 'ts': now_ts}
                prev_date_final = prev_date

        today_top = records_today[:limit]
        prev_top = records_prev[:limit]
        prev_codes = {r['code'] for r in prev_top}
        new_list = [r for r in today_top if r['code'] not in prev_codes]

        return jsonify({'date': curr_date_final, 'prev_date': prev_date_final, 'limit': limit, 'new': new_list})
    except Exception as e:
        return jsonify({'date': '', 'prev_date': '', 'limit': 0, 'new': [], 'message': str(e)})

@app.route('/api/stocks/top/streak')
def get_top_streak():
    try:
        limit = int(request.args.get('limit', '50'))
        k = int(request.args.get('k', '2'))
        date_arg = request.args.get('date')
        ts_mod = _load_tushare()
        token = os.getenv('TUSHARE_TOKEN')
        pro = ts_mod.pro_api(token) if token else ts_mod.pro_api()
        if date_arg:
            end_date = date_arg
            end_compact = _date_to_compact(date_arg)
        else:
            today = datetime.now().strftime('%Y%m%d')
            cal = pro.trade_cal(exchange='SSE', start_date='20000101', end_date=today, is_open=1)
            cal = cal.sort_values(by='cal_date')
            open_dates = cal['cal_date'].tolist()
            d = open_dates[-1] if open_dates else today
            end_compact = d
            end_date = _compact_to_date(d)
        _sync_calendar(pro, '20000101', end_compact)
        dates_k = _get_open_dates_from_db(end_compact, k)
        if len(dates_k) < k:
            dates_k = _get_open_dates(pro, end_compact, k)
        if not dates_k:
            return jsonify({'date': end_date, 'k': k, 'limit': limit, 'list': []})
        name_map = _stock_name_map()
        top_sets: List[set] = []
        last_top: List[Dict[str, Any]] = []
        for i, d in enumerate(dates_k):
            d_date = _compact_to_date(d)
            top_db = _get_top_from_db(d_date, limit)
            if top_db:
                top = top_db
            else:
                df = pro.daily(trade_date=d)
                if df.empty:
                    top = []
                else:
                    recs = _build_daily_records_from_df(df, name_map, _compact_to_date(d))
                    _upsert_rankings(_compact_to_date(d), recs)
                    top = recs[:limit]
            codes = {r['code'] for r in top}
            top_sets.append(codes)
            if i == len(dates_k) - 1:
                last_top = top
        if not top_sets:
            return jsonify({'date': end_date, 'k': k, 'limit': limit, 'list': []})
        common = set.intersection(*top_sets)
        result = [r for r in last_top if r['code'] in common]
        return jsonify({'date': _compact_to_date(dates_k[-1]), 'k': k, 'limit': limit, 'list': result})
    except Exception as e:
        return jsonify({'date': '', 'k': 0, 'limit': 0, 'list': [], 'message': str(e)})

@app.route('/api/stocks/history')
def get_stock_history():
    try:
        code = request.args.get('code')
        days = int(request.args.get('days', '30'))
        date_arg = request.args.get('date')
        if not code:
            return jsonify({'code': '', 'days': days, 'history': []})
        ts_mod = _load_tushare()
        token = os.getenv('TUSHARE_TOKEN')
        pro = ts_mod.pro_api(token) if token else ts_mod.pro_api()
        if date_arg:
            end_date = date_arg
            end_compact = _date_to_compact(date_arg)
        else:
            today = datetime.now().strftime('%Y%m%d')
            cal = pro.trade_cal(exchange='SSE', start_date='20000101', end_date=today, is_open=1)
            cal = cal.sort_values(by='cal_date')
            open_dates = cal['cal_date'].tolist()
            d = open_dates[-1] if open_dates else today
            end_compact = d
            end_date = _compact_to_date(d)
        _sync_calendar(pro, '20000101', end_compact)
        dates_k = _get_open_dates_from_db(end_compact, days)
        if len(dates_k) < days:
            dates_k = _get_open_dates(pro, end_compact, days)
        dates_k = dates_k
        conn = _get_db_conn()
        history: List[Dict[str, Any]] = []
        if conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT trade_date AS "date", amount, rank
                    FROM stock_rankings
                    WHERE code = %s AND trade_date BETWEEN %s::date - (%s::int - 1) * INTERVAL '1 day' AND %s::date
                    ORDER BY trade_date ASC
                    """,
                    (code, _compact_to_date(dates_k[-1]), days, _compact_to_date(dates_k[-1])),
                )
                rows = cur.fetchall()
                if rows:
                    history = rows
        if not history:
            name_map = _stock_name_map()
            for d in dates_k:
                df = pro.daily(trade_date=d)
                if df.empty:
                    continue
                recs = _build_daily_records_from_df(df, name_map, _compact_to_date(d))
                _upsert_rankings(_compact_to_date(d), recs)
                idx_map = {r['code']: i + 1 for i, r in enumerate(recs)}
                r = next((x for x in recs if x['code'] == code), None)
                if r:
                    history.append({'date': r['date'], 'amount': r['amount'], 'rank': idx_map.get(code)})
            history = sorted(history, key=lambda x: x['date'])
        return jsonify({'code': code, 'days': days, 'history': history})
    except Exception as e:
        return jsonify({'code': '', 'days': 0, 'history': [], 'message': str(e)})

if __name__ == '__main__':
    _ensure_db_schema()
    port = int(os.getenv('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=False)
