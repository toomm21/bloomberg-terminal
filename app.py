# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit "Bloomberg"-Dashboard mit:
# - Echte Kurse: Refinitiv Eikon (falls verfügbar) oder yfinance (Fallback)
# - Echte News: yfinance Ticker-News + RSS (Reuters, CNBC)
# - Echter Economic Calendar: Financial Modeling Prep (FMP) API (free, demo key möglich)
# - Robust gegen None/empty, vereinheitlichte Spalten, schwarzes UI
# ─────────────────────────────────────────────────────────────────────────────

import os
from datetime import datetime, timedelta
import time
import requests
import feedparser

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── App-Konfig ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bloomberg Terminal",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Keys/Secrets ─────────────────────────────────────────────────────────────
EIKON_APP_KEY = st.secrets.get("EIKON_APP_KEY") or os.getenv("EIKON_APP_KEY")
FMP_API_KEY   = st.secrets.get("FMP_API_KEY") or os.getenv("FMP_API_KEY") or "demo"  # 'demo' funktioniert eingeschränkt

# ── Eikon Setup (optional) ───────────────────────────────────────────────────
try:
    import eikon as ek
    if EIKON_APP_KEY:
        ek.set_app_key(EIKON_APP_KEY)
        EIKON_AVAILABLE = True
    else:
        EIKON_AVAILABLE = False
except Exception as e:
    EIKON_AVAILABLE = False
    st.warning(f"Eikon API nicht aktiv: {e}")

# ── yfinance Fallback ────────────────────────────────────────────────────────
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception as e:
    YFINANCE_AVAILABLE = False
    st.error(f"yfinance fehlt: {e}")

# ── CSS (kompakt) ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color:#000!important; }
    .main .block-container{ background-color:#000!important; color:#fff!important; padding:3px!important; max-width:none!important; font-family:'Courier New',monospace!important; font-size:10px!important;}
    #MainMenu, footer, header, .stDeployButton{ visibility:hidden; }
    .stTextInput > div > div > input{ background:#1a1a1a!important; color:#fff!important; border:1px solid #333!important; font-family:'Courier New',monospace!important; font-size:9px!important; height:20px!important; padding:2px 4px!important;}
    .stButton > button{ background:#FF8C00!important; color:#000!important; border:none!important; font-family:'Courier New',monospace!important; font-size:6px!important; font-weight:bold!important; padding:0 2px!important; margin:0!important; height:14px!important; min-height:14px!important; width:auto!important; min-width:18px!important; line-height:1!important;}
    .stButton > button:hover{ background:#FFB84D!important; }
    .bloomberg-header{ background:linear-gradient(90deg,#FF8C00 0%,#FFB84D 100%); color:#000; font-weight:bold; font-size:9px; padding:2px 5px; margin-bottom:2px; font-family:'Courier New',monospace;}
    .panel-header{ background:#FF8C00; color:#000; padding:1px 3px; margin-bottom:2px; font-weight:bold; font-size:8px; font-family:'Courier New',monospace;}
    .data-panel{ background:#0A0A0A; border:1px solid #333; padding:3px; font-size:8px; font-family:'Courier New',monospace; color:#FFF; margin:1px 0;}
    .price-up{ color:#00FF00; } .price-down{ color:#FF4444; } .price-neutral{ color:#FFFF00; } .price-white{ color:#FFFFFF; }
    .compact-row{ display:flex; justify-content:space-between; padding:1px 0; border-bottom:1px solid #1a1a1a; font-size:7px; }
    .chart-container{ background:#0A0A0A; border:1px solid #333; padding:2px; margin:1px 0;}
    .status-bar{ background:#FF8C00; color:#000; padding:2px 5px; font-size:8px; position:fixed; bottom:0; left:0; right:0; z-index:999; font-family:'Courier New',monospace; font-weight:bold;}
    p,div,span{ margin:0!important; padding:0!important; line-height:1.1!important;}
    .stMarkdown{ margin:0!important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL.O', 'MSFT.O', 'GOOGL.O', 'TSLA.O', 'AMZN.O']
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'AAPL.O'
if 'chart_timeframe' not in st.session_state:
    st.session_state.chart_timeframe = '1D'

# ── Datenfunktionen: Quotes ─────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_quote(ric: str):
    # EIKON
    if EIKON_AVAILABLE:
        try:
            hist_df = ek.get_timeseries(
                ric,
                fields=['CLOSE', 'HIGH', 'LOW', 'VOLUME'],
                start_date=datetime.now() - timedelta(days=252),
                end_date=datetime.now()
            )
            if hist_df is not None and not hist_df.empty:
                hist_df.columns = [c.upper() for c in hist_df.columns]
                current_price = float(hist_df['CLOSE'].iloc[-1])
                prev_close    = float(hist_df['CLOSE'].iloc[-2]) if len(hist_df) > 1 else current_price
                price_1mo_ago = float(hist_df['CLOSE'].iloc[-21]) if len(hist_df) > 21 else current_price
                price_3mo_ago = float(hist_df['CLOSE'].iloc[-63]) if len(hist_df) > 63 else current_price
                price_ytd     = float(hist_df['CLOSE'].iloc[0])

                change     = current_price - prev_close
                change_pct = (change/prev_close)*100 if prev_close else 0
                pct_1mo    = (current_price/price_1mo_ago-1)*100 if price_1mo_ago else 0
                pct_3mo    = (current_price/price_3mo_ago-1)*100 if price_3mo_ago else 0
                pct_ytd    = (current_price/price_ytd-1)*100 if price_ytd else 0

                volume = int(hist_df['VOLUME'].iloc[-1]) if 'VOLUME' in hist_df.columns else 0
                high   = float(hist_df['HIGH'].iloc[-1]) if 'HIGH' in hist_df.columns else current_price
                low    = float(hist_df['LOW'].iloc[-1]) if 'LOW' in hist_df.columns else current_price

                return {'symbol': ric, 'name': ric, 'price': round(current_price,2),
                        'change': round(change,2), 'change_pct': round(change_pct,2),
                        'pct_1mo': round(pct_1mo,1), 'pct_3mo': round(pct_3mo,1), 'pct_ytd': round(pct_ytd,1),
                        'volume': volume, 'high': round(high,2), 'low': round(low,2)}
        except Exception as e:
            st.warning(f"Eikon error for {ric}: {e}")

    # yfinance
    if YFINANCE_AVAILABLE:
        try:
            symbol = ric.replace('.OQ','').replace('.O','').replace('.N','')
            hist = yf.Ticker(symbol).history(period="1y")
            if hist is not None and not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                prev_close    = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                price_1mo_ago = float(hist['Close'].iloc[-21]) if len(hist) > 21 else current_price
                price_3mo_ago = float(hist['Close'].iloc[-63]) if len(hist) > 63 else current_price
                price_ytd     = float(hist['Close'].iloc[0])

                change     = current_price - prev_close
                change_pct = (change/prev_close)*100 if prev_close else 0
                pct_1mo    = (current_price/price_1mo_ago-1)*100 if price_1mo_ago else 0
                pct_3mo    = (current_price/price_3mo_ago-1)*100 if price_3mo_ago else 0
                pct_ytd    = (current_price/price_ytd-1)*100 if price_ytd else 0

                return {'symbol': ric, 'name': symbol, 'price': round(current_price,2),
                        'change': round(change,2), 'change_pct': round(change_pct,2),
                        'pct_1mo': round(pct_1mo,1), 'pct_3mo': round(pct_3mo,1), 'pct_ytd': round(pct_ytd,1),
                        'volume': int(hist['Volume'].iloc[-1]),
                        'high': round(float(hist['High'].iloc[-1]),2),
                        'low': round(float(hist['Low'].iloc[-1]),2)}
        except Exception as e:
            st.warning(f"yfinance error for {ric}: {e}")

    # Demo fallback
    return {'symbol': ric,'name': ric,'price': 100.00,'change': 0.50,'change_pct': 0.5,
            'pct_1mo': 2.5,'pct_3mo': 5.0,'pct_ytd': 10.0,'volume': 1_000_000,'high': 101.00,'low': 99.00}

# ── Datenfunktionen: Timeseries ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_timeseries(ric: str, timeframe: str='1D') -> pd.DataFrame:
    # EIKON
    if EIKON_AVAILABLE:
        try:
            interval_map = {'1D':1,'5D':5,'1M':30,'3M':90,'6M':180,'1Y':365}
            start_date = datetime.now() - timedelta(days=interval_map.get(timeframe,1))
            df = ek.get_timeseries(
                ric, fields=['OPEN','HIGH','LOW','CLOSE','VOLUME'],
                start_date=start_date, end_date=datetime.now()
            )
            if df is not None and not df.empty:
                df.columns = [c.upper() for c in df.columns]
                df = df.reset_index().rename(columns={'Date':'datetime'})
                if 'datetime' not in df.columns:
                    # manchmal 'DATE' o.ä.
                    cand = [c for c in df.columns if c.lower() in ('date','datetime')]
                    if cand: df = df.rename(columns={cand[0]:'datetime'})
                df = df.rename(columns={'OPEN':'open','HIGH':'high','LOW':'low','CLOSE':'close','VOLUME':'volume'})
                return df[['datetime','open','high','low','close','volume']]
        except Exception as e:
            st.warning(f"Eikon timeseries error for {ric}: {e}")

    # yfinance
    if YFINANCE_AVAILABLE:
        try:
            symbol = ric.replace('.OQ','').replace('.O','').replace('.N','')
            period_map   = {'1D':'1d','5D':'5d','1M':'1mo','3M':'3mo','6M':'6mo','1Y':'1y'}
            interval_map = {'1D':'5m','5D':'15m','1M':'1h','3M':'1d','6M':'1d','1Y':'1wk'}
            hist = yf.Ticker(symbol).history(period=period_map.get(timeframe,'1d'),
                                             interval=interval_map.get(timeframe,'5m'))
            if hist is not None and not hist.empty:
                hist = hist.reset_index()
                dtcol = [c for c in hist.columns if str(c).lower() in ('datetime','date','index')]
                if dtcol: hist = hist.rename(columns={dtcol[0]:'datetime'})
                hist.columns = [str(c).lower() for c in hist.columns]
                return hist[['datetime','open','high','low','close','volume']]
        except Exception as e:
            st.warning(f"yfinance timeseries error for {ric}: {e}")

    # Demo fallback
    periods = 48
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
    price = 100.0
    data = []
    for d in dates:
        price += np.random.normal(0, 0.5)
        data.append({'datetime': d,'open': price,'high': price + abs(np.random.normal(0,0.3)),
                     'low': price - abs(np.random.normal(0,0.3)),'close': price,'volume': np.random.randint(10_000,100_000)})
    return pd.DataFrame(data)

# ── ECHTE NEWS ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=180)
def get_market_news_live(max_items: int = 15) -> list[dict]:
    """Reuters + CNBC RSS (öffentlich), zusammengeführt und chronologisch sortiert."""
    feeds = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC Top News & Analysis
    ]
    items = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:max_items]:
                ts = None
                # verschiedene Zeitfelder robust abfangen
                if hasattr(e, "published_parsed") and e.published_parsed:
                    ts = datetime(*e.published_parsed[:6])
                elif hasattr(e, "updated_parsed") and e.updated_parsed:
                    ts = datetime(*e.updated_parsed[:6])
                items.append({
                    "time": ts.isoformat() if ts else "",
                    "headline": e.title if hasattr(e, "title") else "",
                    "source": "Reuters" if "reuters" in url else "CNBC",
                    "link": e.link if hasattr(e, "link") else ""
                })
        except Exception:
            continue
    # sortieren (neu zuerst)
    items.sort(key=lambda x: x["time"], reverse=True)
    return items[:max_items]

@st.cache_data(ttl=180)
def get_ticker_news(ric: str, max_items: int = 10) -> list[dict]:
    """Ticker-bezogene News per yfinance (kostenlos)."""
    out = []
    if not YFINANCE_AVAILABLE:
        return out
    try:
        symbol = ric.replace('.OQ','').replace('.O','').replace('.N','')
        news = yf.Ticker(symbol).news or []
        for n in news[:max_items]:
            # yfinance liefert epoch-seconds in providerPublishTime
            ts = n.get("providerPublishTime")
            dt = datetime.fromtimestamp(ts) if ts else None
            out.append({
                "time": dt.strftime("%Y-%m-%d %H:%M") if dt else "",
                "headline": n.get("title",""),
                "source": n.get("publisher",""),
                "link": n.get("link","")
            })
    except Exception:
        pass
    return out

# ── ECHTER ECONOMIC CALENDAR (FMP) ───────────────────────────────────────────
@st.cache_data(ttl=300)
def get_economic_calendar_live(days_ahead: int = 7) -> pd.DataFrame:
    """Financial Modeling Prep (free). Nutzt FMP_API_KEY (secrets) oder 'demo'."""
    start = datetime.utcnow().date()
    end   = start + timedelta(days=days_ahead)
    url = (
        f"https://financialmodelingprep.com/api/v3/economic_calendar"
        f"?from={start.isoformat()}&to={end.isoformat()}&apikey={FMP_API_KEY}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame(columns=["date","country","event","actual","previous","change","estimate","impact"])
        df = pd.DataFrame(data)
        # Spalten vereinheitlichen/Subset
        keep = [c for c in ["date","country","event","actual","previous","change","estimate","impact"] if c in df.columns]
        df = df[keep]
        # Darstellung hübscher
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # impact/spalten als kurze Marker
        return df.sort_values("date", ascending=True)
    except Exception as e:
        st.warning(f"Calendar-Fehler (FMP): {e}")
        return pd.DataFrame(columns=["date","country","event","actual","previous","change","estimate","impact"])

# ── Header ───────────────────────────────────────────────────────────────────
current_time = datetime.now().strftime("%H:%M:%S")
data_source = "EIKON" if EIKON_AVAILABLE else "yfinance" if YFINANCE_AVAILABLE else "simulated"
session_status = "ACTIVE" if (EIKON_AVAILABLE or YFINANCE_AVAILABLE) else "OFFLINE"

st.markdown(f"""
<div class="bloomberg-header">
    BLOOMBERG TERMINAL | Market Data & Analytics | {current_time} CET
    <span style="float: right;">Data: {data_source.upper()} | Session: {session_status}</span>
</div>
""", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────────────────────
col_left, col_main, col_right = st.columns([1, 3, 1])

# LEFT COLUMN ─────────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="panel-header">Symbol Selector</div>', unsafe_allow_html=True)
    new_symbol = st.text_input("", value=st.session_state.selected_symbol, key="symbol_input", placeholder="RIC Code").upper()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ Chart", use_container_width=True):
            st.session_state.selected_symbol = new_symbol
            st.rerun()
    with c2:
        if st.button("➕ Add", use_container_width=True):
            if new_symbol and new_symbol not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_symbol)
                st.rerun()

    st.markdown('<div class="panel-header">Quick Access</div>', unsafe_allow_html=True)
    quick_symbols = ['.SPX', '.IXIC', '.DJI', 'AAPL.O', 'MSFT.O', 'TSLA.O']
    cols1 = st.columns(3)
    for i, symbol in enumerate(quick_symbols[:3]):
        with cols1[i]:
            if st.button(symbol[:5], key=f"q1_{symbol}"):
                st.session_state.selected_symbol = symbol
                st.rerun()
    cols2 = st.columns(3)
    for i, symbol in enumerate(quick_symbols[3:]):
        with cols2[i]:
            if st.button(symbol[:5], key=f"q2_{symbol}"):
                st.session_state.selected_symbol = symbol
                st.rerun()

    st.markdown('<div class="panel-header">Watchlist</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="data-panel">
        <div style="display:flex; font-size:6px; font-weight:bold; color:#FF8C00; border-bottom:1px solid #333; padding-bottom:1px; margin-bottom:1px;">
            <span style="width:35px;">Ticker</span>
            <span style="width:40px;">Last</span>
            <span style="width:30px;">%Chg</span>
            <span style="width:30px;">%1Mo</span>
            <span style="width:30px;">%3Mo</span>
            <span style="width:30px;">%YTD</span>
            <span style="width:25px;"></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for symbol in st.session_state.watchlist[:]:
        q = get_quote(symbol)
        chg_color = "price-up" if q['change_pct'] >= 0 else "price-down"
        mo1_color = "price-up" if q['pct_1mo'] >= 0 else "price-down"
        mo3_color = "price-up" if q['pct_3mo'] >= 0 else "price-down"
        ytd_color = "price-up" if q['pct_ytd'] >= 0 else "price-down"

        c_l, c_r = st.columns([4,1])
        with c_l:
            ticker_display = symbol[:6]
            st.markdown(
                f'<div style="display:flex; font-size:6px; align-items:center; padding:0px; border-bottom:1px solid #1a1a1a;">'
                f'<span style="width:35px; color:white;">{ticker_display}</span>'
                f'<span style="width:40px; color:white;">{q["price"]:.2f}</span>'
                f'<span style="width:30px;" class="{chg_color}">{q["change_pct"]:+.1f}%</span>'
                f'<span style="width:30px;" class="{mo1_color}">{q["pct_1mo"]:+.1f}%</span>'
                f'<span style="width:30px;" class="{mo3_color}">{q["pct_3mo"]:+.1f}%</span>'
                f'<span style="width:30px;" class="{ytd_color}">{q["pct_ytd"]:+.1f}%</span>'
                f'</div>', unsafe_allow_html=True
            )
        with c_r:
            b1, b2 = st.columns(2)
            with b1:
                if st.button("📈", key=f"ch_{symbol}", help="Chart"):
                    st.session_state.selected_symbol = symbol
                    st.rerun()
            with b2:
                if st.button("🗑️", key=f"rm_{symbol}", help="Remove"):
                    st.session_state.watchlist.remove(symbol)
                    st.rerun()

# MAIN COLUMN ────────────────────────────────────────────────────────────────
with col_main:
    current_symbol = st.session_state.selected_symbol
    quote = get_quote(current_symbol)

    tf_cols = st.columns([2,1,1,1,1,1,1])
    with tf_cols[0]:
        st.markdown(f'<div style="font-size:8px; color:white; padding-top:2px;">{current_symbol} Chart:</div>', unsafe_allow_html=True)
    for i, tf in enumerate(['1D','5D','1M','3M','6M','1Y']):
        with tf_cols[i+1]:
            if st.button(tf, key=f"tf_{tf}"):
                st.session_state.chart_timeframe = tf
                st.rerun()

    color = "price-up" if quote['change'] >= 0 else "price-down"
    st.markdown(f"""
    <div class="chart-container">
      <div style="background:#FF8C00; color:#000; padding:1px 3px; font-size:8px; font-weight:bold;">
        {current_symbol} | {quote['name'][:20]} | Last: {quote['price']:.2f} | 
        Chg: <span class="{color}">{quote['change']:+.2f} ({quote['change_pct']:+.2f}%)</span> |
        Vol: {quote['volume']:,} | Range: {quote['low']:.2f}-{quote['high']:.2f} | TF: {st.session_state.chart_timeframe}
      </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        chart_data = get_timeseries(current_symbol, st.session_state.chart_timeframe)
        if not chart_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=chart_data['datetime'], open=chart_data['open'], high=chart_data['high'],
                low=chart_data['low'], close=chart_data['close'], name=current_symbol,
                increasing_line_color='#00FF00', decreasing_line_color='#FF4444',
                increasing_fillcolor='rgba(0,255,0,0.1)', decreasing_fillcolor='rgba(255,68,68,0.1)'
            ))
            fig.update_layout(plot_bgcolor='#000', paper_bgcolor='#000', font=dict(color='white', size=7),
                              height=280, showlegend=False, margin=dict(l=35, r=15, t=5, b=25),
                              xaxis=dict(gridcolor='#333', showgrid=True, color='white'),
                              yaxis=dict(gridcolor='#333', showgrid=True, color='white'),
                              xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown(f"""
        <div class="chart-container">
          <div style="background:#FF8C00; color:#000; padding:1px 3px; font-size:8px; font-weight:bold;">
            Perf | 1Mo: <span class="{'price-up' if quote['pct_1mo'] >= 0 else 'price-down'}">{quote['pct_1mo']:+.1f}%</span> | 
            3Mo: <span class="{'price-up' if quote['pct_3mo'] >= 0 else 'price-down'}">{quote['pct_3mo']:+.1f}%</span> | 
            YTD: <span class="{'price-up' if quote['pct_ytd'] >= 0 else 'price-down'}">{quote['pct_ytd']:+.1f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not chart_data.empty:
            fig_vol = go.Figure()
            colors = ['#00FF00' if c >= o else '#FF4444' for c, o in zip(chart_data['close'], chart_data['open'])]
            fig_vol.add_trace(go.Bar(x=chart_data['datetime'], y=chart_data['volume'], marker_color=colors, name='Volume'))
            fig_vol.update_layout(plot_bgcolor='#000', paper_bgcolor='#000', font=dict(color='white', size=7),
                                  height=130, showlegend=False, margin=dict(l=35, r=15, t=5, b=25),
                                  xaxis=dict(gridcolor='#333', showgrid=True, color='white'),
                                  yaxis=dict(gridcolor='#333', showgrid=True, color='white', title='Vol'))
            st.plotly_chart(fig_vol, use_container_width=True, config={'displayModeBar': False})
    except Exception as e:
        st.error(f"Chart Error: {e}")

# RIGHT COLUMN ───────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="panel-header">Economic Calendar</div>', unsafe_allow_html=True)
    eco = get_economic_calendar_live(days_ahead=7)
    st.markdown('<div class="data-panel">', unsafe_allow_html=True)
    if eco.empty:
        st.info("Keine Kalenderdaten erhalten (FMP). Prüfe API Key oder Internet.")
    else:
        # begrenzt darstellen
        for _, row in eco.head(12).iterrows():
            t = row['date']
            t_str = t.strftime("%Y-%m-%d %H:%M") if pd.notna(t) else ""
            impact = row.get("impact", "")
            color = "price-neutral"
            if isinstance(impact, str):
                if impact.lower() in ("high","3"): color = "price-down"
                elif impact.lower() in ("medium","2"): color = "price-neutral"
                elif impact.lower() in ("low","1"): color = "price-up"
            st.markdown(f"""
            <div class="compact-row">
              <span>{t_str}</span>
              <span class="{color}">{row.get('country','')}</span>
            </div>
            <div style="font-size:6px; margin-bottom:2px;">
              {row.get('event','')}
              <br><span class="price-neutral">A:{row.get('actual','—')} | F:{row.get('estimate','—')} | P:{row.get('previous','—')}</span>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-header">Market News</div>', unsafe_allow_html=True)
    market_news = get_market_news_live(max_items=10)
    st.markdown('<div class="data-panel">', unsafe_allow_html=True)
    if not market_news:
        st.info("Keine Markt-News geladen.")
    else:
        for n in market_news:
            t = n.get("time","")
            src = n.get("source","")
            head = n.get("headline","")
            st.markdown(f"""
            <div class="compact-row">
              <span class="price-neutral">{t[11:16] if len(t)>=16 else ''}</span>
              <span style="font-size:6px;">{src}</span>
            </div>
            <div style="font-size:7px; margin-bottom:2px; line-height:1.2;">
              <a href="{n.get('link','')}" target="_blank" style="color:#fff; text-decoration:none;">{head}</a>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-header">Ticker News</div>', unsafe_allow_html=True)
    tnews = get_ticker_news(st.session_state.selected_symbol, max_items=8)
    st.markdown('<div class="data-panel">', unsafe_allow_html=True)
    if not tnews:
        st.info("Keine Ticker-News gefunden.")
    else:
        for n in tnews:
            st.markdown(f"""
            <div class="compact-row">
              <span class="price-neutral">{n.get('time','')}</span>
              <span style="font-size:6px;">{n.get('source','')}</span>
            </div>
            <div style="font-size:7px; margin-bottom:2px; line-height:1.2;">
              <a href="{n.get('link','')}" target="_blank" style="color:#fff; text-decoration:none;">{n.get('headline','')}</a>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Status Bar ─────────────────────────────────────────────────────────────────
current_time = datetime.now().strftime("%H:%M:%S")
st.markdown(f"""
<div class="status-bar">
    TERMINAL CONNECTED | Symbol: {st.session_state.selected_symbol} |
    Watchlist: {len(st.session_state.watchlist)} | Time: {current_time} |
    Data: {'EIKON LIVE' if EIKON_AVAILABLE else 'YFINANCE' if YFINANCE_AVAILABLE else 'DEMO'}
    <span style="float:right;">User: TRADER</span>
</div>
""", unsafe_allow_html=True)