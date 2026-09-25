"""
PRO FOOTBALL PREDICTOR - Streamlit Version
Hybrid Model + ELO + 5 AI Agents + Weather + Tables
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import requests as req
from scipy.stats import poisson

# ============ Page Config ============
st.set_page_config(
    page_title="Pro Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============ RTL CSS + Custom Styling ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Cairo', 'Segoe UI', Tahoma, sans-serif !important;
        direction: rtl;
        text-align: right;
    }
    
    .main { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 8px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        color: #475569;
        font-family: 'Cairo', sans-serif;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e40af, #3b82f6) !important;
        color: white !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-family: 'Cairo', sans-serif;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
    }
    
    .stSelectbox label, .stSlider label {
        font-family: 'Cairo', sans-serif;
        font-weight: 600;
        color: #0f172a;
    }
    
    /* Match Cards */
    .match-card {
        background: white;
        border-radius: 16px;
        padding: 18px;
        margin: 12px 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        direction: rtl;
        font-family: 'Cairo', 'Segoe UI', Tahoma;
        color: #0f172a;
    }
    
    .match-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 2px solid #f1f5f9;
    }
    
    .league-badge {
        background: #dbeafe;
        color: #1e40af;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: bold;
    }
    
    .date-badge {
        background: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: bold;
        margin-right: 8px;
    }
    
    .match-grid {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 12px;
        align-items: center;
        margin: 16px 0;
    }
    
    .team-name {
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
    }
    
    .team-elo {
        font-size: 10px;
        color: #64748b;
    }
    
    .score-display {
        font-size: 32px;
        font-weight: 900;
        color: #1e40af;
        letter-spacing: 2px;
    }
    
    .prob-bar {
        display: flex;
        height: 28px;
        border-radius: 8px;
        overflow: hidden;
        font-size: 11px;
        font-weight: bold;
        color: white;
        line-height: 28px;
        text-align: center;
        margin: 14px 0;
    }
    
    .prob-home { background: linear-gradient(90deg, #10b981, #059669); }
    .prob-draw { background: linear-gradient(90deg, #94a3b8, #64748b); }
    .prob-away { background: linear-gradient(90deg, #ef4444, #dc2626); }
    
    .agent-box {
        border-radius: 10px;
        padding: 10px;
        margin: 6px 0;
        border-left: 4px solid;
        direction: rtl;
    }
    
    .meta-agent {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        border-radius: 12px;
        padding: 14px;
        margin-top: 10px;
        color: white;
        text-align: center;
    }
    
    .badge-w { background: #10b981; color: white; }
    .badge-d { background: #f59e0b; color: white; }
    .badge-l { background: #ef4444; color: white; }
    
    .form-badge {
        display: inline-block;
        width: 18px;
        height: 18px;
        line-height: 18px;
        text-align: center;
        border-radius: 50%;
        color: white;
        font-size: 10px;
        font-weight: bold;
        margin: 1px;
    }
    
    /* Tables */
    .standings-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        color: #0f172a;
        direction: rtl;
    }
    
    .standings-table th {
        background: #1e40af;
        color: white;
        padding: 10px 6px;
        text-align: center;
        font-size: 11px;
    }
    
    .standings-table td {
        padding: 8px 6px;
        border-bottom: 1px solid #e2e8f0;
        text-align: center;
    }
    
    .standings-table tr:nth-child(even) { background: #f8fafc; }
    
    .fixtures-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        color: #0f172a;
        direction: rtl;
    }
    
    .fixtures-table th {
        background: #1e40af;
        color: white;
        padding: 10px 6px;
    }
    
    .fixtures-table td {
        padding: 8px 6px;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .fixtures-table tr:nth-child(even) { background: #f8fafc; }
</style>
""", unsafe_allow_html=True)


# ============ Load Data (cached) ============
@st.cache_resource
def load_all():
    with open('model_data.json', 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    dc_params = model_data['dc_params']
    final_elo = {k: float(v) for k, v in model_data['elo_final'].items()}
    HOME_ADV = float(model_data['home_adv'])
    RHO = float(model_data['rho'])
    FEATURES = model_data['feature_cols']
    
    with open('hybrid_model.pkl', 'rb') as f:
        cal = pickle.load(f)
    
    matches_df = pd.read_csv('matches.csv')
    matches_df['date_parsed'] = pd.to_datetime(matches_df['date_parsed'], errors='coerce', utc=True)
    matches_df = matches_df.dropna(subset=['date_parsed']).reset_index(drop=True)
    
    upcoming = pd.read_csv('upcoming.csv')
    upcoming['date_parsed'] = pd.to_datetime(upcoming['date'], errors='coerce', utc=True)
    upcoming = upcoming.dropna(subset=['date_parsed']).sort_values('date_parsed').reset_index(drop=True)
    
    return dc_params, final_elo, HOME_ADV, RHO, FEATURES, cal, matches_df, upcoming


dc_params, final_elo, HOME_ADV, RHO, FEATURES, cal, matches_df, upcoming = load_all()


# ============ Build Indexes ============
@st.cache_resource
def build_indexes(matches_df):
    h_arr = matches_df['home_team'].values
    a_arr = matches_df['away_team'].values
    hs_arr = matches_df['home_score'].values.astype(int)
    as_arr = matches_df['away_score'].values.astype(int)
    dates_ns = matches_df['date_parsed'].astype('int64').values
    
    team_history = {}
    h2h_index = {}
    for i in range(len(matches_df)):
        h, a = h_arr[i], a_arr[i]
        hs, as_ = hs_arr[i], as_arr[i]
        d_ns = int(dates_ns[i])
        res_h = 'W' if hs > as_ else ('D' if hs == as_ else 'L')
        team_history.setdefault(h, []).append((i, True, hs, as_, res_h, a, d_ns))
        res_a = 'W' if as_ > hs else ('D' if as_ == hs else 'L')
        team_history.setdefault(a, []).append((i, False, as_, hs, res_a, h, d_ns))
        key = tuple(sorted([h, a]))
        h2h_index.setdefault(key, []).append((i, h, int(hs), int(as_), d_ns))
    return team_history, h2h_index, h_arr, a_arr, hs_arr, as_arr, dates_ns


team_history, h2h_index, h_arr, a_arr, hs_arr, as_arr, dates_ns = build_indexes(matches_df)


# ============ Helper Functions ============
def to_ns(date_val):
    try:
        ts = pd.Timestamp(date_val)
        if ts.tz is None:
            ts = ts.tz_localize('UTC')
        return int(ts.value)
    except:
        return int(pd.Timestamp.now(tz='UTC').value)


def form_feats(team, cutoff_ns, n=5):
    hist = team_history.get(team, [])
    relevant = []
    for row in reversed(hist):
        if row[6] < cutoff_ns:
            relevant.append(row)
            if len(relevant) >= n:
                break
    if not relevant:
        return {'pts': 1.0, 'gf': 1.0, 'ga': 1.0, 'played': 0, 'results': []}
    pts, gf, ga, results = 0, 0, 0, []
    for row in relevant:
        gf += row[2]; ga += row[3]
        if row[4] == 'W': pts += 3
        elif row[4] == 'D': pts += 1
        results.append(row[4])
    n_act = len(relevant)
    return {'pts': pts/n_act, 'gf': gf/n_act, 'ga': ga/n_act, 'played': n_act, 'results': results[::-1]}


def h2h_feats(h, a, cutoff_ns, n=5):
    key = tuple(sorted([h, a]))
    hist = h2h_index.get(key, [])
    relevant = []
    for row in reversed(hist):
        if row[4] < cutoff_ns:
            relevant.append(row)
            if len(relevant) >= n:
                break
    if not relevant:
        return {'hw': 0.33, 'd': 0.34, 'aw': 0.33, 'matches': 0}
    hw, aw = 0, 0
    for row in relevant:
        idx, home_t, hs, as_, d = row
        if home_t == h:
            if hs > as_: hw += 1
            elif as_ > hs: aw += 1
        else:
            if as_ > hs: hw += 1
            elif hs > as_: aw += 1
    total = len(relevant)
    d = total - hw - aw
    return {'hw': hw/total, 'd': d/total, 'aw': aw/total, 'matches': total}


def rest_days(team, cutoff_ns):
    hist = team_history.get(team, [])
    for row in reversed(hist):
        if row[6] < cutoff_ns:
            diff_ns = cutoff_ns - row[6]
            return min(diff_ns / (86400 * 1e9), 60)
    return 7.0


def predict_full(h, a, is_neutral, date_str):
    if h not in dc_params['attack'] or a not in dc_params['attack']:
        return None
    cutoff_ns = to_ns(date_str)
    hf = form_feats(h, cutoff_ns)
    af = form_feats(a, cutoff_ns)
    h2h = h2h_feats(h, a, cutoff_ns)
    h_e = final_elo.get(h, 1500)
    a_e = final_elo.get(a, 1500)
    
    fv = np.array([[
        h_e, a_e, h_e-a_e,
        dc_params['attack'].get(h, 0), dc_params['attack'].get(a, 0),
        dc_params['defense'].get(h, 0), dc_params['defense'].get(a, 0),
        hf['pts'], af['pts'], hf['gf'], af['gf'], hf['ga'], af['ga'],
        h2h['hw'], h2h['d'], h2h['aw'],
        rest_days(h, cutoff_ns), rest_days(a, cutoff_ns),
    ]])
    
    proba = cal.predict_proba(fv)[0]
    pa, pd_, ph = float(proba[0]), float(proba[1]), float(proba[2])
    
    a_h = dc_params['attack'].get(h, 0); a_a = dc_params['attack'].get(a, 0)
    d_h = dc_params['defense'].get(h, 0); d_a = dc_params['defense'].get(a, 0)
    ha = 0 if is_neutral else HOME_ADV
    lh = float(np.exp(min(a_h + d_a + ha, 2.5)))
    la = float(np.exp(min(a_a + d_h, 2.5)))
    
    mg = 8
    M = np.zeros((mg, mg))
    for i in range(mg):
        for j in range(mg):
            p = poisson.pmf(i, lh) * poisson.pmf(j, la)
            if i == 0 and j == 0: p *= max(1 - lh*la*RHO, 1e-6)
            elif i == 0 and j == 1: p *= max(1 + lh*RHO, 1e-6)
            elif i == 1 and j == 0: p *= max(1 + la*RHO, 1e-6)
            elif i == 1 and j == 1: p *= max(1 - RHO, 1e-6)
            M[i, j] = max(p, 1e-12)
    M /= M.sum()
    
    Mh = sum(M[i, j] for i in range(mg) for j in range(mg) if i > j)
    Md = sum(M[i, i] for i in range(mg))
    Ma = 1 - Mh - Md
    if Mh > 0 and Ma > 0 and Md > 0:
        for i in range(mg):
            for j in range(mg):
                if i > j: M[i, j] *= ph / Mh
                elif i < j: M[i, j] *= pa / Ma
                else: M[i, j] *= pd_ / Md
        M /= M.sum()
    
    flat = sorted([(M[i, j], i, j) for i in range(mg) for j in range(mg)], reverse=True)
    return {'ph': ph, 'pd': pd_, 'pa': pa, 'best': flat[0], 'top5': flat[:5],
            'lh': lh, 'la': la, 'h_elo': h_e, 'a_elo': a_e,
            'h_form': hf, 'a_form': af, 'h2h': h2h}


# ============ Weather ============
STADIUM_COORDS = {
    'Morocco': (34.0209, -6.8416, 'Stade Moulay Abdellah'),
    'Senegal': (14.7289, -17.1949, 'Stade Abdoulaye Wade'),
    'Egypt': (30.0690, 31.3125, 'Cairo Stadium'),
    'Nigeria': (5.0397, 7.9206, 'Godswill Akpabio'),
    'Algeria': (36.7367, 3.0869, 'Stade 5 Juillet'),
    'Tunisia': (36.7481, 10.2731, 'Stade de Rades'),
    'Ghana': (5.5629, -0.1955, 'Accra Sports'),
    'South Africa': (-26.1328, 27.9826, 'FNB Stadium'),
    'England': (51.5560, -0.2795, 'Wembley'),
    'France': (48.9245, 2.3601, 'Stade de France'),
    'Spain': (40.4530, -3.6883, 'Bernabeu'),
    'Germany': (48.2188, 11.6247, 'Allianz Arena'),
    'Italy': (41.9338, 12.4547, 'Olimpico'),
    'Portugal': (38.7528, -9.1850, 'Da Luz'),
    'Netherlands': (52.3144, 4.9416, 'Cruijff ArenA'),
    'Belgium': (50.8947, 4.3335, 'Baudouin'),
    'Werder Bremen': (53.0662, 8.8376, 'Weserstadion'),
    'Borussia Dortmund': (51.4926, 7.4518, 'Signal Iduna Park'),
    'FC Bayern Munchen': (48.2188, 11.6247, 'Allianz Arena'),
    'Bayern': (48.2188, 11.6247, 'Allianz Arena'),
    'RB Leipzig': (51.3457, 12.3485, 'Red Bull Arena'),
    'Bayer Leverkusen': (51.0383, 7.0025, 'BayArena'),
    'Eintracht Frankfurt': (50.0686, 8.6454, 'Deutsche Bank Park'),
    'VfB Stuttgart': (48.7924, 9.2320, 'Mercedes-Benz Arena'),
    'Arsenal': (51.5549, -0.1084, 'Emirates'),
    'Liverpool FC': (53.4308, -2.9608, 'Anfield'),
    'Manchester City': (53.4831, -2.2004, 'Etihad'),
    'Manchester United': (53.4631, -2.2913, 'Old Trafford'),
    'Chelsea': (51.4815, -0.1910, 'Stamford Bridge'),
    'Tottenham': (51.6043, -0.0665, 'Tottenham Stadium'),
    'Newcastle United': (54.9756, -1.6217, 'St. James Park'),
    'Aston Villa': (52.5092, -1.8847, 'Villa Park'),
    'FC Barcelona': (41.3809, 2.1228, 'Camp Nou'),
    'Real Madrid': (40.4530, -3.6883, 'Bernabeu'),
    'Atletico Madrid': (40.4362, -3.5995, 'Metropolitano'),
    'Inter': (45.4780, 9.1240, 'San Siro'),
    'AC Milan': (45.4780, 9.1240, 'San Siro'),
    'Juventus': (45.1096, 7.6411, 'Allianz Stadium'),
    'Napoli': (40.8279, 14.1930, 'Maradona'),
    'Paris Saint-Germain': (48.8414, 2.2530, 'Parc des Princes'),
    'PSV Eindhoven': (51.4417, 5.4685, 'Philips Stadion'),
    'Feyenoord': (51.8939, 4.5234, 'De Kuip'),
    'Ajax': (52.3144, 4.9416, 'Cruijff ArenA'),
    'FC Porto': (41.1618, -8.5836, 'Dragao'),
    'Benfica': (38.7528, -9.1850, 'Da Luz'),
    'Sporting CP': (38.7611, -9.1608, 'Alvalade'),
}

_weather_cache = {}


@st.cache_data(ttl=3600)
def get_weather_cached(home, date_str):
    key = f"{home}_{date_str[:10]}"
    if key in _weather_cache:
        return _weather_cache[key]
    coords = STADIUM_COORDS.get(home)
    if not coords:
        _weather_cache[key] = None
        return None
    lat, lng, stadium = coords
    try:
        r = req.get("https://api.open-meteo.com/v1/forecast", params={
            'latitude': lat, 'longitude': lng,
            'start_date': date_str[:10], 'end_date': date_str[:10],
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m',
            'timezone': 'auto'
        }, timeout=6)
        if r.status_code != 200:
            _weather_cache[key] = None
            return None
        h = r.json().get('hourly', {})
        w = {'temp': float(np.mean(h.get('temperature_2m', [20]))),
             'humidity': float(np.mean(h.get('relative_humidity_2m', [50]))),
             'precip': float(np.sum(h.get('precipitation', [0]))),
             'wind': float(np.mean(h.get('wind_speed_10m', [10]))),
             'stadium': stadium, 'lat': lat, 'lng': lng}
        _weather_cache[key] = w
        return w
    except Exception:
        _weather_cache[key] = None
        return None


# ============ Agents ============
def agent_stat(p, h, a):
    if p['ph'] > p['pd'] and p['ph'] > p['pa']:
        v = f"فوز {h}"
    elif p['pa'] > p['ph'] and p['pa'] > p['pd']:
        v = f"فوز {a}"
    else:
        v = "تعادل"
    return {'v': v, 'c': max(p['ph'], p['pd'], p['pa']),
            'r': f"Hybrid: {p['ph']*100:.0f}%/{p['pd']*100:.0f}%/{p['pa']*100:.0f}%"}


def agent_tech(p, h, a):
    d = p['h_form']['pts'] - p['a_form']['pts']
    if d > 0.5:
        v, c = f"فوز {h}", min(0.6 + d*0.1, 0.85)
    elif d < -0.5:
        v, c = f"فوز {a}", min(0.6 + abs(d)*0.1, 0.85)
    else:
        v, c = "متكافئة", 0.5
    return {'v': v, 'c': c, 'r': f"{h}:{p['h_form']['pts']:.1f} | {a}:{p['a_form']['pts']:.1f}"}


def agent_env(w):
    if not w:
        return {'v': 'غير متاح', 'c': 0.5, 'r': 'لا بيانات طقس'}
    imp = 0; r = []
    if w['precip'] > 5: imp -= 0.1; r.append(f"أمطار {w['precip']:.1f}مم")
    elif w['precip'] > 1: imp -= 0.05; r.append("أمطار خفيفة")
    if w['wind'] > 30: imp -= 0.08; r.append(f"رياح {w['wind']:.0f}كم/س")
    if w['temp'] > 32: imp -= 0.06; r.append(f"حرارة {w['temp']:.0f}")
    elif w['temp'] < 3: imp -= 0.05; r.append(f"برد {w['temp']:.0f}")
    if not r: r = ["طقس مثالي"]
    return {'v': 'محايد' if imp == 0 else f"{imp*100:+.0f}%",
            'c': 0.7 if imp == 0 else 0.6, 'r': ' | '.join(r)}


def agent_critic(s, t, e):
    w = []
    if s['v'].startswith('فوز') and t['v'].startswith('فوز'):
        if s['v'].split('فوز ')[-1] != t['v'].split('فوز ')[-1]:
            w.append("الاحصائي والفني مختلفان")
    if 'محايد' not in e['v'] and e['v'] != 'غير متاح':
        w.append(f"{e['v']}")
    conf = (s['c'] + t['c'] + e['c']) / 3
    return {'v': f"{len(w)} تحذير" if w else "اجماع",
            'c': conf*0.7 if w else min(conf*1.1, 0.9),
            'r': ' | '.join(w) if w else "الكل متفقون"}


def agent_meta(s, t, e, c, p, h, a):
    v = {'home': p['ph']*0.45, 'draw': p['pd']*0.45, 'away': p['pa']*0.45}
    ts = 0.3 * t['c']
    if 'فوز' in t['v']:
        if h in t['v']: v['home'] += ts
        else: v['away'] += ts
    else:
        v['draw'] += ts
    tot = sum(v.values())
    fp = {k: x/tot for k, x in v.items()}
    best = max(fp, key=fp.get)
    vm = {'home': f"فوز {h}", 'draw': "تعادل", 'away': f"فوز {a}"}
    return {'fp': fp, 'v': vm[best], 'c': fp[best]}


def badge_html(res):
    colors = {'W': '#10b981', 'D': '#f59e0b', 'L': '#ef4444'}
    return ''.join([f'<span class="form-badge" style="background:{colors.get(x, "#666")};">{x}</span>' for x in res])


# ============ Render Match Card ============
def render_match_card(row):
    try:
        h, a = row['home_team'], row['away_team']
        if h not in dc_params['attack'] or a not in dc_params['attack']:
            return None
        is_neutral = bool(row.get('is_neutral', False)) if pd.notna(row.get('is_neutral')) else False
        date_str = str(row['date'])[:10]
        p = predict_full(h, a, is_neutral, row['date'])
        if not p:
            return None
        w = get_weather_cached(h, date_str)
        s = agent_stat(p, h, a)
        t = agent_tech(p, h, a)
        e = agent_env(w)
        c = agent_critic(s, t, e)
        m = agent_meta(s, t, e, c, p, h, a)
        best = p['best']
        
        prob_bar = f'''<div class="prob-bar">
            <div class="prob-home" style="width:{p['ph']*100}%;">{p['ph']*100:.0f}%</div>
            <div class="prob-draw" style="width:{p['pd']*100}%;">{p['pd']*100:.0f}%</div>
            <div class="prob-away" style="width:{p['pa']*100}%;">{p['pa']*100:.0f}%</div>
        </div>'''
        
        chips = ''
        for prob, i, j in p['top5'][:3]:
            hl = 'background:#3b82f6;color:white;' if (i, j) == (best[1], best[2]) else 'background:#f1f5f9;color:#0f172a;'
            chips += f'<span style="display:inline-block;{hl}padding:6px 12px;border-radius:8px;margin:2px;font-size:13px;font-weight:bold;">{i}-{j} ({prob*100:.1f}%)</span>'
        
        h2h = p['h2h']
        if h2h['matches'] > 0:
            hw = int(h2h['hw']*h2h['matches']); aw = int(h2h['aw']*h2h['matches']); dr = h2h['matches'] - hw - aw
            h2h_html = f'''<div style="display:flex;justify-content:space-between;font-size:12px;font-weight:bold;">
                <span style="color:#10b981;">{hw} {h}</span>
                <span style="color:#64748b;">{dr} تعادل</span>
                <span style="color:#ef4444;">{aw} {a}</span></div>'''
        else:
            h2h_html = '<div style="text-align:center;color:#94a3b8;font-size:11px;">لا مواجهات</div>'
        
        weather_html = '<div style="text-align:center;color:#64748b;font-size:11px;">غير متاح</div>'
        if w:
            alert = 'طقس مثالي'
            if w['precip'] > 5: alert = 'امطار غزيرة'
            elif w['wind'] > 30: alert = 'رياح قوية'
            elif w['temp'] > 32: alert = 'حرارة مرتفعة'
            elif w['temp'] < 3: alert = 'برد قارس'
            weather_html = f'''<div style="font-size:11px;color:#475569;line-height:1.6;">
                <div>{w['temp']:.0f}C | {w['humidity']:.0f}%</div>
                <div>{w['wind']:.0f}كم/س | {w['precip']:.1f}مم</div>
                <div style="color:#1e40af;text-align:center;margin-top:4px;font-weight:bold;">{alert}</div>
                <div style="text-align:center;color:#3b82f6;font-size:10px;">{w['stadium']}</div>
            </div>'''
        
        def agent_box(name, icon, ag, color):
            cn = ag['c']*100
            bc = "#10b981" if cn >= 60 else ("#f59e0b" if cn >= 40 else "#ef4444")
            return f'''<div class="agent-box" style="background:{color};border-color:{bc};">
                <div style="display:flex;justify-content:space-between;">
                    <b style="font-size:12px;color:#0f172a;">{icon} {name}</b>
                    <b style="font-size:11px;color:{bc};">{cn:.0f}%</b>
                </div>
                <div style="font-size:11px;margin:4px 0;color:#0f172a;"><b>{ag['v']}</b></div>
                <div style="font-size:10px;color:#475569;">{ag['r']}</div>
                <div style="height:4px;background:#e2e8f0;border-radius:2px;margin-top:6px;">
                    <div style="width:{cn}%;height:100%;background:{bc};"></div>
                </div>
            </div>'''
        
        meta_html = f'''<div class="meta-agent">
            <div style="font-size:13px;font-weight:bold;">Meta-Agent</div>
            <div style="font-size:18px;font-weight:900;margin:6px 0;">{m['v']}</div>
            <div style="font-size:12px;opacity:0.9;">ثقة: {m['c']*100:.0f}%</div>
            <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin-top:8px;">
                <div style="width:{m['fp']['home']*100}%;background:#10b981;"></div>
                <div style="width:{m['fp']['draw']*100}%;background:#94a3b8;"></div>
                <div style="width:{m['fp']['away']*100}%;background:#ef4444;"></div>
            </div>
        </div>'''
        
        return f'''
        <div class="match-card">
            <div class="match-header">
                <div>
                    <span class="league-badge">{row['league']}</span>
                    <span class="date-badge">{date_str}</span>
                </div>
                <span style="color:#64748b;font-size:11px;">{'محايد' if is_neutral else 'ارض'}</span>
            </div>
            
            <div class="match-grid">
                <div style="text-align:center;">
                    <div class="team-name">{h}</div>
                    <div class="team-elo">ELO {p['h_elo']:.0f}</div>
                    <div style="margin-top:6px;">{badge_html(p['h_form']['results'])}</div>
                </div>
                <div style="text-align:center;">
                    <div class="score-display">{best[1]} - {best[2]}</div>
                    <div style="font-size:10px;color:#64748b;">الاكثر احتمالا</div>
                    <div style="font-size:12px;font-weight:bold;color:#3b82f6;">{best[0]*100:.1f}%</div>
                </div>
                <div style="text-align:center;">
                    <div class="team-name">{a}</div>
                    <div class="team-elo">ELO {p['a_elo']:.0f}</div>
                    <div style="margin-top:6px;">{badge_html(p['a_form']['results'])}</div>
                </div>
            </div>
            
            {prob_bar}
            
            <div style="margin:12px 0;text-align:center;">{chips}</div>
            
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px 0;">
                <div style="background:#f8fafc;border-radius:10px;padding:10px;">
                    <div style="font-size:11px;font-weight:bold;color:#475569;text-align:center;margin-bottom:6px;">الاهداف</div>
                    <div style="text-align:center;font-size:12px;color:#0f172a;">
                        <div>شوط1: <b>{p['lh']*0.43:.1f}-{p['la']*0.43:.1f}</b></div>
                        <div>شوط2: <b>{p['lh']*0.57:.1f}-{p['la']*0.57:.1f}</b></div>
                        <div style="color:#1e40af;">المجموع: <b>{p['lh']:.1f}-{p['la']:.1f}</b></div>
                    </div>
                </div>
                <div style="background:#f0f9ff;border-radius:10px;padding:10px;">
                    <div style="font-size:11px;font-weight:bold;color:#475569;text-align:center;margin-bottom:6px;">الطقس</div>
                    {weather_html}
                </div>
            </div>
            
            <div style="background:#fffbeb;border-radius:10px;padding:10px;">
                <div style="font-size:11px;font-weight:bold;color:#92400e;text-align:center;margin-bottom:4px;">H2H</div>
                {h2h_html}
            </div>
            
            <div style="background:#f8fafc;border-radius:12px;padding:12px;margin-top:12px;">
                <div style="font-size:12px;font-weight:bold;color:#475569;text-align:center;margin-bottom:8px;">الوكلاء الخمسة</div>
                {agent_box('الاحصائي', '📊', s, '#dbeafe')}
                {agent_box('الفني', '⚽', t, '#dcfce7')}
                {agent_box('البيئي', '🌤️', e, '#fef3c7')}
                {agent_box('الناقد', '🔍', c, '#fee2e2')}
                {meta_html}
            </div>
        </div>'''
    except Exception as ex:
        return None


# ============ Standings (Current Season) ============
def get_current_season_start(league_name):
    lg_matches = matches_df[matches_df['league'] == league_name]
    if len(lg_matches) == 0:
        return None
    latest = lg_matches['date_parsed'].max()
    euro_leagues = ['League', 'Liga', 'Serie', 'Bundesliga', 'Ligue', 'Eredivisie', 'Championship']
    if any(k in league_name for k in euro_leagues):
        if latest.month >= 7:
            season_start = pd.Timestamp(f"{latest.year}-08-01", tz='UTC')
        else:
            season_start = pd.Timestamp(f"{latest.year - 1}-08-01", tz='UTC')
    else:
        season_start = latest - pd.Timedelta(days=300)
    return season_start, latest


def compute_standings_current(league_name):
    result = get_current_season_start(league_name)
    if result is None:
        return None
    season_start, latest = result
    lg_matches = matches_df[matches_df['league'] == league_name]
    season_matches = lg_matches[lg_matches['date_parsed'] >= season_start]
    if len(season_matches) < 3:
        season_matches = lg_matches[lg_matches['date_parsed'] >= season_start - pd.Timedelta(days=60)]
    if len(season_matches) == 0:
        return None
    
    stats = {}
    for _, m in season_matches.iterrows():
        h, a = m['home_team'], m['away_team']
        hs, as_ = int(m['home_score']), int(m['away_score'])
        for team, gf, ga in [(h, hs, as_), (a, as_, hs)]:
            if team not in stats:
                stats[team] = {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'Pts': 0}
            stats[team]['P'] += 1
            stats[team]['GF'] += gf
            stats[team]['GA'] += ga
            if gf > ga: stats[team]['W'] += 1; stats[team]['Pts'] += 3
            elif gf == ga: stats[team]['D'] += 1; stats[team]['Pts'] += 1
            else: stats[team]['L'] += 1
    
    rows = []
    for team, s in stats.items():
        rows.append({
            'الفريق': team, 'لعب': s['P'], 'فاز': s['W'], 'تعادل': s['D'], 'خسر': s['L'],
            'له': s['GF'], 'عليه': s['GA'], 'الفارق': s['GF'] - s['GA'], 'النقاط': s['Pts'],
        })
    df_s = pd.DataFrame(rows)
    df_s = df_s.sort_values(['النقاط', 'الفارق', 'له'], ascending=False).reset_index(drop=True)
    df_s['#'] = range(1, len(df_s) + 1)
    return df_s, len(season_matches), season_start, latest


def render_standings_html(league):
    result = compute_standings_current(league)
    if result is None:
        return '<div style="text-align:center;padding:40px;color:#64748b;">لا توجد بيانات كافية</div>'
    df_s, n_matches, season_start, latest = result
    
    if latest.month >= 7:
        season_label = f"{latest.year}/{latest.year + 1}"
    else:
        season_label = f"{latest.year - 1}/{latest.year}"
    
    rows_html = ""
    for i, r in df_s.iterrows():
        pos = r['#']
        if pos <= 4: side = "#10b981"
        elif pos <= 6: side = "#3b82f6"
        elif pos >= len(df_s) - 2: side = "#ef4444"
        else: side = "transparent"
        gd = r['الفارق']
        gd_color = "#10b981" if gd > 0 else ("#ef4444" if gd < 0 else "#64748b")
        
        rows_html += f'''<tr>
            <td style="border-right:3px solid {side};font-weight:bold;color:#0f172a;">{pos}</td>
            <td style="text-align:right;font-weight:600;color:#0f172a;white-space:nowrap;">{r['الفريق']}</td>
            <td style="color:#475569;">{r['لعب']}</td>
            <td style="color:#10b981;font-weight:bold;">{r['فاز']}</td>
            <td style="color:#f59e0b;">{r['تعادل']}</td>
            <td style="color:#ef4444;">{r['خسر']}</td>
            <td style="color:#475569;">{r['له']}</td>
            <td style="color:#475569;">{r['عليه']}</td>
            <td style="color:{gd_color};font-weight:bold;">{gd:+d}</td>
            <td style="color:#1e40af;font-weight:900;font-size:13px;">{r['النقاط']}</td>
        </tr>'''
    
    return f'''
    <div class="match-card">
        <h2 style="text-align:center;color:#0f172a;margin:0 0 6px 0;font-size:18px;">ترتيب {league}</h2>
        <div style="text-align:center;color:#64748b;font-size:11px;margin-bottom:10px;">
            موسم {season_label} | {n_matches} مباراة
        </div>
        <div style="overflow-x:auto;">
            <table class="standings-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th style="text-align:right;">الفريق</th>
                        <th>لعب</th>
                        <th>فاز</th>
                        <th>تعادل</th>
                        <th>خسر</th>
                        <th>له</th>
                        <th>عليه</th>
                        <th>+/-</th>
                        <th>نقاط</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        <div style="display:flex;gap:10px;justify-content:center;margin-top:10px;font-size:10px;color:#64748b;">
            <span><span style="display:inline-block;width:8px;height:8px;background:#10b981;border-radius:2px;"></span> أبطال</span>
            <span><span style="display:inline-block;width:8px;height:8px;background:#3b82f6;border-radius:2px;"></span> أوروبي</span>
            <span><span style="display:inline-block;width:8px;height:8px;background:#ef4444;border-radius:2px;"></span> هبوط</span>
        </div>
    </div>'''


def render_fixtures_html(league, max_n=20):
    f = upcoming.copy()
    if league != "الكل":
        f = f[f['league'] == league]
    if len(f) == 0:
        return f'<div style="text-align:center;padding:40px;color:#64748b;">لا توجد مباريات قادمة</div>'
    f = f.head(int(max_n))
    
    rows_html = ""
    for i, r in f.iterrows():
        h, a = r['home_team'], r['away_team']
        date_str = str(r['date'])[:10]
        try: time_str = str(r['date'])[11:16]
        except: time_str = ""
        neutral = '🏟️' if r.get('is_neutral', False) else '🏠'
        rows_html += f'''<tr>
            <td style="color:#64748b;font-size:11px;">{date_str}</td>
            <td style="text-align:right;font-weight:bold;color:#0f172a;font-size:12px;">{h}</td>
            <td style="text-align:center;font-size:11px;">{neutral}</td>
            <td style="text-align:left;font-weight:bold;color:#0f172a;font-size:12px;">{a}</td>
            <td style="color:#64748b;font-size:11px;">{time_str}</td>
        </tr>'''
    
    return f'''
    <div class="match-card">
        <h2 style="text-align:center;color:#0f172a;margin:0 0 12px 0;font-size:18px;">المباريات القادمة</h2>
        <div style="overflow-x:auto;">
            <table class="fixtures-table">
                <thead>
                    <tr>
                        <th>التاريخ</th>
                        <th style="text-align:right;">المستضيف</th>
                        <th></th>
                        <th style="text-align:left;">الضيف</th>
                        <th>الوقت</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    </div>'''


# ============ Main App ============
st.markdown('''
<div style="background:linear-gradient(135deg,#1e40af,#3b82f6);border-radius:18px;padding:24px;text-align:center;color:white;margin-bottom:16px;">
    <div style="font-size:28px;font-weight:900;">⚽ PRO FOOTBALL PREDICTOR</div>
    <div style="font-size:13px;opacity:0.9;margin-top:6px;">Hybrid + ELO + 5 AI Agents + Weather</div>
    <div style="font-size:11px;opacity:0.8;margin-top:4px;">16,377 matches | 488 teams | 15 tournaments</div>
</div>
''', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🎯 التوقعات", "📅 جدول المباريات", "🏆 الترتيب", "📊 ترتيب ELO"])

leagues_list = ["الكل"] + sorted(upcoming['league'].dropna().unique().tolist())
standings_leagues = sorted(matches_df['league'].dropna().unique().tolist())

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        lg = st.selectbox("🏆 البطولة", leagues_list, index=leagues_list.index("UNL") if "UNL" in leagues_list else 1, key="lg1")
    with col2:
        nn = st.slider("عدد المباريات", 1, 15, 3, key="nn1")
    
    if st.button("🎯 اعرض التوقعات", key="btn1"):
        f = upcoming.copy()
        if lg != "الكل": f = f[f['league'] == lg]
        f = f.head(nn)
        for _, r in f.iterrows():
            card = render_match_card(r)
            if card:
                st.markdown(card, unsafe_allow_html=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        lg2 = st.selectbox("🏆 البطولة", leagues_list, index=leagues_list.index("UNL") if "UNL" in leagues_list else 1, key="lg2")
    with col2:
        nn2 = st.slider("عدد المباريات", 5, 40, 20, key="nn2")
    
    if st.button("📅 اعرض الجدول", key="btn2"):
        st.markdown(render_fixtures_html(lg2, nn2), unsafe_allow_html=True)

with tab3:
    lg3 = st.selectbox("🏆 البطولة", standings_leagues, index=standings_leagues.index("La_Liga") if "La_Liga" in standings_leagues else 0, key="lg3")
    if st.button("🏆 اعرض الترتيب", key="btn3"):
        st.markdown(render_standings_html(lg3), unsafe_allow_html=True)

with tab4:
    tn = st.slider("عدد الفرق", 10, 40, 20, key="tn4")
    if st.button("📊 اعرض الترتيب", key="btn4"):
        attacks = sorted(dc_params['attack'].items(), key=lambda x: -x[1])[:tn]
        html = '''<div class="match-card"><table class="standings-table">
        <thead><tr><th>#</th><th style="text-align:right;">الفريق</th><th>الهجوم</th><th>الدفاع</th><th>ELO</th></tr></thead><tbody>'''
        for i, (t, v) in enumerate(attacks, 1):
            d = dc_params['defense'].get(t, 0)
            elo = final_elo.get(t, 1500)
            html += f'''<tr>
                <td style="color:#1e40af;font-weight:bold;">{i}</td>
                <td style="text-align:right;font-weight:600;color:#0f172a;">{t}</td>
                <td style="color:#059669;font-weight:bold;">{v:+.2f}</td>
                <td style="color:#dc2626;font-weight:bold;">{d:+.2f}</td>
                <td style="color:#7c3aed;font-weight:bold;">{elo:.0f}</td>
            </tr>'''
        html += '</tbody></table></div>'
        st.markdown(html, unsafe_allow_html=True)
