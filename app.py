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
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_rankings_date_rank
            ON stock_rankings (trade_date, rank);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_rankings_code_date
            ON stock_rankings (code, trade_date);
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_theme (
              code text PRIMARY KEY,
              name text,
              theme text,
              updated_at timestamptz DEFAULT now()
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
            SELECT code,
                   name,
                   price::float8 AS price,
                   change::float8 AS change,
                   change_percent::float8 AS "changePercent",
                   volume,
                   amount,
                   to_char(trade_date, 'YYYY-MM-DD') AS "date",
                   rank
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
        db_only_arg = request.args.get('db_only', '1')
        db_only = db_only_arg.lower() in ('1', 'true', 'yes')
        conn = _get_db_conn()
        if not conn:
            return jsonify({'date': '', 'k': 0, 'limit': 0, 'list': [], 'message': '数据库连接失败'}), 500
        if date_arg:
            end_date = date_arg
        else:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(trade_date) FROM stock_rankings")
                row = cur.fetchone()
                end_date = row[0] if row and row[0] else None
            if not end_date:
                return jsonify({'date': '', 'k': k, 'limit': limit, 'list': [], 'message': 'need_ingest'}), 200
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT trade_date
                FROM stock_rankings
                WHERE trade_date <= %s
                ORDER BY trade_date DESC
                LIMIT %s
                """,
                (end_date, k),
            )
            date_rows = cur.fetchall()
        if not date_rows or len(date_rows) < k:
            end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
            return jsonify({'date': end_str, 'k': k, 'limit': limit, 'list': [], 'message': 'need_ingest'}), 200
        dates_list = [dr[0] for dr in reversed(date_rows)]
        placeholders = ','.join(['%s'] * len(dates_list))
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT code
                FROM stock_rankings
                WHERE trade_date IN ({placeholders}) AND rank <= %s
                GROUP BY code
                HAVING COUNT(DISTINCT trade_date) = %s
                """,
                (*dates_list, limit, k),
            )
            rows = cur.fetchall()
            codes = [r[0] for r in rows]
        if not codes:
            last_date = dates_list[-1]
            last_date_str = last_date.strftime('%Y-%m-%d') if hasattr(last_date, 'strftime') else str(last_date)
            return jsonify({'date': last_date_str, 'k': k, 'limit': limit, 'list': []})
        codes_placeholders = ','.join(['%s'] * len(codes))
        last_date = dates_list[-1]
        last_date_str = last_date.strftime('%Y-%m-%d') if hasattr(last_date, 'strftime') else str(last_date)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT code,
                       name,
                       price::float8 AS price,
                       change::float8 AS change,
                       change_percent::float8 AS "changePercent",
                       volume,
                       amount,
                       to_char(trade_date, 'YYYY-MM-DD') AS "date",
                       rank
                FROM stock_rankings
                WHERE trade_date = %s AND rank <= %s AND code IN ({codes_placeholders})
                ORDER BY rank ASC
                """,
                (last_date, limit, *codes),
            )
            result_rows = cur.fetchall()
        return jsonify({'date': last_date_str, 'k': k, 'limit': limit, 'list': result_rows})
    except Exception as e:
        return jsonify({'date': '', 'k': 0, 'limit': 0, 'list': [], 'message': str(e)})

@app.route('/api/stocks/history')
def get_stock_history():
    try:
        code = request.args.get('code')
        days = int(request.args.get('days', '30'))
        date_arg = request.args.get('date')
        if not code:
            return jsonify({'code': '', 'name': '', 'days': days, 'history': []})
        conn = _get_db_conn()
        if not conn:
            return jsonify({'code': code, 'name': '', 'days': days, 'history': [], 'message': '数据库连接失败'}), 500
        if date_arg:
            end_date = date_arg
        else:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(trade_date) FROM stock_rankings")
                row = cur.fetchone()
                end_date = row[0] if row and row[0] else None
            if not end_date:
                return jsonify({'code': code, 'name': '', 'days': days, 'history': [], 'message': 'need_ingest'}), 200
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT trade_date
                FROM stock_rankings
                WHERE trade_date <= %s
                ORDER BY trade_date DESC
                LIMIT %s
                """,
                (end_date, days),
            )
            date_rows = cur.fetchall()
        if not date_rows:
            return jsonify({'code': code, 'name': '', 'days': days, 'history': [], 'message': 'need_ingest'}), 200
        dates_list = [dr[0] for dr in reversed(date_rows)]
        placeholders = ','.join(['%s'] * len(dates_list))
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT to_char(trade_date, 'YYYY-MM-DD') AS "date",
                       amount,
                       rank,
                       code,
                       name
                FROM stock_rankings
                WHERE code = %s AND trade_date IN ({placeholders})
                ORDER BY trade_date ASC
                """,
                (code, *dates_list),
            )
            rows = cur.fetchall()
        # 固定最近交易日序列，缺失填充 amount=0, rank=NULL
        seq = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in dates_list]
        row_map = {r['date']: r for r in rows}
        name_val = rows[0]['name'] if rows else ''
        history = [
            {
                'date': dstr,
                'amount': int(row_map.get(dstr, {}).get('amount', 0)),
                'rank': row_map.get(dstr, {}).get('rank', None)
            }
            for dstr in seq
        ]
        return jsonify({'code': code, 'name': name_val, 'days': days, 'history': history})
    except Exception as e:
        return jsonify({'code': '', 'name': '', 'days': 0, 'history': [], 'message': str(e)})

@app.route('/api/admin/ingest', methods=['GET', 'POST'])
def ingest_bulk():
    try:
        conn = _get_db_conn()
        if not conn:
            return jsonify({'message': '数据库连接失败: 未检测到有效的 DATABASE_URL', 'exchange': '', 'start': '', 'end': '', 'days': 0, 'rows': 0}), 500
        start = request.args.get('start')
        end = request.args.get('end')
        limit = int(request.args.get('limit', '200'))
        exchange = request.args.get('exchange', 'SSE')
        if not start or not end:
            return jsonify({'message': '缺少 start/end 参数'}), 400
        ts_mod = _load_tushare()
        token = os.getenv('TUSHARE_TOKEN')
        pro = ts_mod.pro_api(token) if token else ts_mod.pro_api()
        start_compact = _date_to_compact(start)
        end_compact = _date_to_compact(end)
        _sync_calendar(pro, start_compact, end_compact)
        cal = pro.trade_cal(exchange=exchange, start_date=start_compact, end_date=end_compact, is_open=1)
        cal = cal.sort_values(by='cal_date')
        open_dates = cal['cal_date'].tolist()
        if not open_dates:
            return jsonify({'exchange': exchange, 'start': start, 'end': end, 'days': 0, 'rows': 0, 'message': '区间内无开放日'})
        name_map = _stock_name_map()
        total_rows = 0
        days_count = 0
        for d in open_dates:
            df = pro.daily(trade_date=d)
            if df.empty:
                continue
            recs = _build_daily_records_from_df(df, name_map, _compact_to_date(d))
            recs_ingest = recs[:limit]
            _upsert_rankings(_compact_to_date(d), recs_ingest)
            total_rows += len(recs_ingest)
            days_count += 1
        return jsonify({'exchange': exchange, 'start': start, 'end': end, 'days': days_count, 'rows': total_rows, 'db_connected': True})
    except Exception as e:
        return jsonify({'message': str(e), 'exchange': '', 'start': '', 'end': '', 'days': 0, 'rows': 0})

@app.route('/api/admin/rankings/stats')
def rankings_stats():
    try:
        start = request.args.get('start')
        end = request.args.get('end')
        if not start or not end:
            return jsonify({'message': '缺少 start/end 参数'}), 400
        conn = _get_db_conn()
        if not conn:
            return jsonify({'message': '数据库连接失败'}), 500
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT to_char(trade_date, 'YYYY-MM-DD') AS date, COUNT(*) AS count
                FROM stock_rankings
                WHERE trade_date BETWEEN %s AND %s
                GROUP BY trade_date
                ORDER BY trade_date
                """,
                (start, end),
            )
            rows = cur.fetchall()
        return jsonify({'start': start, 'end': end, 'stats': rows})
    except Exception as e:
        return jsonify({'message': str(e), 'start': '', 'end': '', 'stats': []})

@app.route('/api/admin/calendar/stats')
def calendar_stats():
    try:
        exchange = request.args.get('exchange', 'SSE')
        conn = _get_db_conn()
        if not conn:
            return jsonify({'message': '数据库连接失败'}), 500
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT to_char(MIN(cal_date), 'YYYY-MM-DD') AS min_date,
                       to_char(MAX(cal_date), 'YYYY-MM-DD') AS max_date,
                       COUNT(*) AS total
                FROM trading_calendar
                WHERE exchange = %s AND is_open = true
                """,
                (exchange,),
            )
            agg = cur.fetchone()
            cur.execute(
                """
                SELECT to_char(cal_date, 'YYYY-MM-DD') AS date
                FROM trading_calendar
                WHERE exchange = %s AND is_open = true
                ORDER BY cal_date DESC
                LIMIT 10
                """,
                (exchange,),
            )
            last10 = cur.fetchall()
        return jsonify({'exchange': exchange, 'aggregate': agg, 'last10': last10})
    except Exception as e:
        return jsonify({'message': str(e), 'exchange': '', 'aggregate': {}, 'last10': []})

@app.route('/api/admin/ingest_recent', methods=['POST', 'GET'])
def ingest_recent():
    try:
        conn = _get_db_conn()
        if not conn:
            return jsonify({'message': '数据库连接失败: 未检测到有效的 DATABASE_URL', 'days': 0, 'rows': 0}), 500
        days = int(request.args.get('days', '7'))
        limit = int(request.args.get('limit', '200'))
        ts_mod = _load_tushare()
        token = os.getenv('TUSHARE_TOKEN')
        pro = ts_mod.pro_api(token) if token else ts_mod.pro_api()
        today = datetime.now().strftime('%Y%m%d')
        cal = pro.trade_cal(exchange='SSE', start_date='20000101', end_date=today, is_open=1)
        cal = cal.sort_values(by='cal_date')
        open_dates = cal['cal_date'].tolist()
        if not open_dates:
            return jsonify({'message': '未获取到交易日历', 'days': 0, 'rows': 0})
        end_compact = open_dates[-1]
        start_idx = max(0, len(open_dates) - days)
        recent_dates = open_dates[start_idx:]
        _sync_calendar(pro, recent_dates[0], recent_dates[-1])
        name_map = _stock_name_map()
        total_rows = 0
        for d in recent_dates:
            df = pro.daily(trade_date=d)
            if df.empty:
                continue
            recs = _build_daily_records_from_df(df, name_map, _compact_to_date(d))
            recs_ingest = recs[:limit]
            _upsert_rankings(_compact_to_date(d), recs_ingest)
            total_rows += len(recs_ingest)
        return jsonify({'days': len(recent_dates), 'rows': total_rows, 'start': _compact_to_date(recent_dates[0]), 'end': _compact_to_date(recent_dates[-1])})
    except Exception as e:
        return jsonify({'message': str(e), 'days': 0, 'rows': 0})

if __name__ == '__main__':
    _ensure_db_schema()
    port = int(os.getenv('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=False)
def _classify_theme_with_gemini(name: str, code: str) -> str:
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return '未知'
        import json
        from urllib.request import Request, urlopen
        from urllib.error import URLError
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
        prompt = (
            "请用一个词输出该股票的题材/板块（例如：白酒、半导体、银行、保险、医药、新能源、军工、地产、汽车、AI等）。"
            f" 股票代码：{code}，名称：{name}。只返回题材名，不要解释。"
        )
        body = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        req = Request(url, data=json.dumps(body).encode('utf-8'), headers={'Content-Type': 'application/json'})
        try:
            resp = urlopen(req, timeout=12)
            data = json.loads(resp.read().decode('utf-8'))
            text = (
                data.get('candidates', [{}])[0]
                    .get('content', {})
                    .get('parts', [{}])[0]
                    .get('text', '')
            )
            theme = (text or '未知').strip().replace('\n', '').replace('：', ':')
            if len(theme) > 20:
                theme = theme[:20]
            return theme or '未知'
        except URLError:
            return '未知'
    except Exception:
        return '未知'

def _get_theme_for(code: str) -> str:
    conn = _get_db_conn()
    if not conn:
        return '未知'
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT theme FROM stock_theme WHERE code = %s", (code,))
        row = cur.fetchone()
        return row['theme'] if row and row.get('theme') else '未知'

def _upsert_theme(code: str, name: str, theme: str):
    conn = _get_db_conn()
    if not conn:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stock_theme (code, name, theme)
            VALUES (%s, %s, %s)
            ON CONFLICT (code) DO UPDATE SET
              name = EXCLUDED.name,
              theme = EXCLUDED.theme,
              updated_at = now()
            """,
            (code, name, theme),
        )

@app.route('/api/ai/classify_theme', methods=['POST'])
def classify_theme():
    try:
        payload = request.get_json(silent=True) or {}
        items = payload.get('items', [])
        results = []
        for it in items:
            code = str(it.get('code', ''))
            name = str(it.get('name', ''))
            if not code:
                continue
            theme = _get_theme_for(code)
            if theme == '未知':
                theme = _classify_theme_with_gemini(name, code)
                _upsert_theme(code, name, theme)
            results.append({ 'code': code, 'name': name, 'theme': theme })
        return jsonify({ 'items': results })
    except Exception as e:
        return jsonify({ 'items': [], 'message': str(e) })

@app.route('/api/stats/themes')
@app.route('/api/stats/themes/')
def theme_stats():
    try:
        date = request.args.get('date')
        limit = int(request.args.get('limit', '200'))
        if not date:
            return jsonify({ 'date': '', 'limit': limit, 'stats': [] })
        conn = _get_db_conn()
        if not conn:
            return jsonify({'date': date, 'limit': limit, 'stats': [], 'message': '数据库连接失败'}), 500
        # 取当日前N名
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT code, name, amount, rank
                FROM stock_rankings
                WHERE trade_date = %s AND rank <= %s
                ORDER BY rank ASC
                """,
                (date, limit),
            )
            rows = cur.fetchall()
        if not rows:
            return jsonify({ 'date': date, 'limit': limit, 'stats': [] })
        # 确保有题材分类
        for r in rows:
            theme = _get_theme_for(r['code'])
            if theme == '未知':
                t = _classify_theme_with_gemini(r.get('name') or '', r['code'])
                _upsert_theme(r['code'], r.get('name') or '', t)
        # 聚合
        conn = _get_db_conn()
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT st.theme,
                       AVG(sr.rank)::float8 AS avg_rank,
                       SUM(sr.amount)::bigint AS total_amount
                FROM stock_rankings sr
                JOIN stock_theme st ON st.code = sr.code
                WHERE sr.trade_date = %s AND sr.rank <= %s
                GROUP BY st.theme
                ORDER BY total_amount DESC
                """,
                (date, limit),
            )
            agg = cur.fetchall()
        total = sum(int(a['total_amount']) for a in agg) or 1
        stats = [
            {
                'theme': a['theme'],
                'avgRank': float(a['avg_rank']),
                'totalAmount': int(a['total_amount']),
                'amountShare': round(int(a['total_amount']) / total * 100, 2)
            }
            for a in agg
        ]
        return jsonify({ 'date': date, 'limit': limit, 'stats': stats })
    except Exception as e:
        return jsonify({ 'date': '', 'limit': 0, 'stats': [], 'message': str(e) })

@app.route('/api/admin/routes')
def list_routes():
    try:
        rules = []
        for r in app.url_map.iter_rules():
            rules.append({
                'rule': str(r),
                'endpoint': r.endpoint,
                'methods': sorted(list(r.methods or []))
            })
        return jsonify({'routes': rules})
    except Exception as e:
        return jsonify({'routes': [], 'message': str(e)})
