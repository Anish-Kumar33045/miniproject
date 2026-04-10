import pdfplumber
import pandas as pd
import numpy as np
import re
import io
from datetime import datetime, timedelta

MERCHANT_RISK = {
    'grocery':0.05,'transport':0.08,'food':0.07,'utility':0.04,
    'shopping':0.18,'entertainment':0.22,'healthcare':0.06,
    'education':0.03,'travel':0.25,'fuel':0.10,'other':0.12
}

MERCHANT_KEYWORDS = {
    'swiggy':'food','zomato':'food','dominos':'food','mcdonalds':'food',
    'kfc':'food','subway':'food','pizza':'food','burger':'food',
    'blinkit':'grocery','bigbasket':'grocery','dmart':'grocery',
    'zepto':'grocery','reliance':'grocery','more supermarket':'grocery',
    'ola':'transport','uber':'transport','rapido':'transport','auto':'transport',
    'irctc':'travel','makemytrip':'travel','goibibo':'travel',
    'cleartrip':'travel','airline':'travel','hotel':'travel',
    'amazon':'shopping','flipkart':'shopping','myntra':'shopping',
    'meesho':'shopping','nykaa':'shopping','ajio':'shopping',
    'electricity':'utility','bescom':'utility','tneb':'utility',
    'water bill':'utility','jio':'utility','airtel':'utility',
    'bsnl':'utility','vi ':'utility','recharge':'utility',
    'apollo':'healthcare','medplus':'healthcare','1mg':'healthcare',
    'practo':'healthcare','pharmacy':'healthcare','hospital':'healthcare',
    'byju':'education','unacademy':'education','coursera':'education',
    'pvr':'entertainment','inox':'entertainment','bookmyshow':'entertainment',
    'netflix':'entertainment','spotify':'entertainment','prime':'entertainment',
    'petrol':'fuel','hp ':'fuel','indian oil':'fuel','bharat petroleum':'fuel',
    'shell':'fuel','fuel':'fuel',
}

BANKS = ['hdfc','sbi','icici','axis','kotak','yes bank','pnb',
         'bob','canara','idfc','rbl','indusind','federal','south indian']
CITIES = ['Mumbai','Delhi','Bengaluru','Chennai','Hyderabad',
          'Pune','Kolkata','Ahmedabad','Jaipur','Surat']


def classify_merchant(desc: str) -> str:
    if not desc:
        return 'other'
    d = desc.lower()
    for kw, cat in MERCHANT_KEYWORDS.items():
        if kw in d:
            return cat
    return 'other'


def detect_bank(desc: str) -> str:
    if not desc:
        return np.random.choice(['HDFC','SBI','ICICI','Axis','Kotak'])
    d = desc.lower()
    for b in BANKS:
        if b in d:
            return b.upper()
    return np.random.choice(['HDFC','SBI','ICICI','Axis','Kotak'])


def parse_amount(s) -> float:
    if s is None:
        return 0.0
    s = re.sub(r'[₹,\s]', '', str(s))
    s = re.sub(r'[DrCrDRCR]+$', '', s, flags=re.IGNORECASE)
    try:
        return abs(float(s))
    except:
        return 0.0


def parse_date(s) -> datetime:
    if not s:
        return None
    s = str(s).strip()
    fmts = [
        '%d/%m/%Y','%d-%m-%Y','%Y-%m-%d','%d/%m/%y','%d-%m-%y',
        '%d %b %Y','%d %B %Y','%b %d, %Y','%B %d, %Y',
        '%m/%d/%Y','%d-%b-%Y','%d %b, %Y','%Y/%m/%d',
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except:
            continue
    # try extracting just the date part
    match = re.search(r'(\d{1,2})[/\-\s](\w+)[/\-\s](\d{2,4})', s)
    if match:
        for fmt in ['%d %b %Y','%d %B %Y','%d/%m/%Y']:
            try:
                cleaned = f"{match.group(1)} {match.group(2)} {match.group(3)}"
                return datetime.strptime(cleaned, '%d %b %Y')
            except:
                continue
    return None


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('txn_date').reset_index(drop=True)
    df['txn_date_dt'] = pd.to_datetime(df['txn_date'], errors='coerce')

    avg7_list, tpd_list = [], []
    for i, row in df.iterrows():
        cutoff = row['txn_date_dt'] - timedelta(days=7)
        past = df[(df['txn_date_dt'] >= cutoff) & (df['txn_date_dt'] < row['txn_date_dt'])]
        avg7 = round(past['amount'].mean(), 2) if len(past) > 0 else row['amount']
        avg7_list.append(avg7)
        same_day = df[df['txn_date'] == row['txn_date']]
        tpd_list.append(len(same_day))

    df['avg_amount_7d']       = avg7_list
    df['txn_per_day']         = tpd_list
    df['amount_to_avg_ratio'] = (df['amount'] / (df['avg_amount_7d'] + 1)).round(3)
    df = df.drop(columns=['txn_date_dt'], errors='ignore')
    return df


def build_row(idx, amount, dt, desc):
    merchant_cat = classify_merchant(desc)
    bank = detect_bank(desc)
    return {
        'txn_id':             f"TXN{idx:06d}",
        'amount':             round(amount, 2),
        'hour':               dt.hour if dt else np.random.randint(8, 22),
        'day_of_week':        dt.weekday() if dt else np.random.randint(0, 7),
        'merchant_cat':       merchant_cat,
        'merchant_risk_score':MERCHANT_RISK.get(merchant_cat, 0.12),
        'is_new_merchant':    0,
        'txn_per_day':        1,
        'avg_amount_7d':      amount,
        'device_change':      0,
        'location_change':    0,
        'failed_txn_count':   0,
        'city':               np.random.choice(CITIES),
        'bank':               bank,
        'is_weekend':         1 if dt and dt.weekday() >= 5 else 0,
        'txn_date':           dt.strftime('%Y-%m-%d') if dt else '2024-01-01',
        'description':        str(desc)[:100],
    }


# ── CSV PARSER ──────────────────────────────────────────────────────────────

def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Handles any CSV format:
    - Our own generated transactions.csv  (has all columns already)
    - GPay CSV export
    - PhonePe CSV export
    - Any generic bank CSV with date + amount columns
    """
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            break
        except:
            continue

    if df is None or df.empty:
        # try skipping rows until we find a real header
        for skip in range(1, 10):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), skiprows=skip,
                                 encoding='utf-8-sig')
                if len(df.columns) >= 2 and len(df) > 0:
                    break
            except:
                continue

    if df is None or df.empty:
        return pd.DataFrame()

    cols_lower = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=cols_lower)
    df.columns = [c.lower().strip() for c in df.columns]

    # ── Case 1: Already our format (has 'amount' and 'merchant_cat') ──
    if 'amount' in df.columns and 'merchant_cat' in df.columns:
        if 'amount_to_avg_ratio' not in df.columns:
            df['amount_to_avg_ratio'] = df['amount'] / (df['avg_amount_7d'] + 1)
        for col, default in [('hour',12),('day_of_week',0),('is_new_merchant',0),
                              ('txn_per_day',1),('avg_amount_7d',500),
                              ('device_change',0),('location_change',0),
                              ('failed_txn_count',0),('is_weekend',0),
                              ('merchant_risk_score',0.1)]:
            if col not in df.columns:
                df[col] = default
        if 'txn_id' not in df.columns:
            df.insert(0,'txn_id',['TXN'+str(i).zfill(6) for i in range(len(df))])
        if 'txn_date' not in df.columns:
            df['txn_date'] = '2024-01-01'
        if 'city' not in df.columns:
            df['city'] = np.random.choice(CITIES, len(df))
        if 'bank' not in df.columns:
            df['bank'] = np.random.choice(['HDFC','SBI','ICICI'], len(df))
        return df

    # ── Case 2: GPay CSV ──
    # Columns: Transaction Date, Transaction ID, Description, Paid/Received (₹), Status
    gpay_date   = next((c for c in df.columns if 'date' in c or 'time' in c), None)
    gpay_amount = next((c for c in df.columns if 'paid' in c or 'received' in c
                        or 'amount' in c or 'amt' in c), None)
    gpay_desc   = next((c for c in df.columns if 'description' in c or 'desc' in c
                        or 'merchant' in c or 'name' in c or 'narration' in c
                        or 'remarks' in c or 'detail' in c or 'to' in c), None)

    if gpay_date and gpay_amount:
        rows = []
        for i, row in df.iterrows():
            amount = parse_amount(row[gpay_amount])
            dt     = parse_date(str(row[gpay_date]))
            desc   = str(row[gpay_desc]) if gpay_desc else ''
            if amount > 0:
                rows.append(build_row(i, amount, dt, desc))
        if rows:
            result = pd.DataFrame(rows)
            return add_engineered_features(result)

    # ── Case 3: Generic — try to find any date+amount columns ──
    date_col   = next((c for c in df.columns if any(k in c for k in
                       ['date','time','txn','transaction','on'])), None)
    amount_col = next((c for c in df.columns if any(k in c for k in
                       ['amount','amt','debit','credit','paid','sum','value','rs','inr'])), None)
    desc_col   = next((c for c in df.columns if any(k in c for k in
                       ['desc','detail','narr','remark','merchant','particulars',
                        'mode','ref','utr','info','note'])), None)

    if date_col and amount_col:
        rows = []
        for i, row in df.iterrows():
            amount = parse_amount(row[amount_col])
            dt     = parse_date(str(row[date_col]))
            desc   = str(row[desc_col]) if desc_col else str(row.to_dict())
            if amount > 0:
                rows.append(build_row(i, amount, dt, desc))
        if rows:
            result = pd.DataFrame(rows)
            return add_engineered_features(result)

    return pd.DataFrame()


# ── PDF PARSER ──────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes) -> pd.DataFrame:
    rows = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            full_text = ''
            for page in pdf.pages:
                # try tables first
                tables = page.extract_tables()
                for table in (tables or []):
                    extracted = _rows_from_table(table)
                    rows.extend(extracted)
                full_text += (page.extract_text() or '') + '\n'

        # if tables gave nothing, parse raw text
        if not rows:
            rows = _rows_from_text(full_text)

    except Exception as e:
        print(f"PDF parse error: {e}")

    if not rows:
        return pd.DataFrame()

    result_rows = []
    for i, r in enumerate(rows):
        amount = parse_amount(r.get('raw_amount', '0'))
        dt     = parse_date(r.get('raw_date', ''))
        desc   = r.get('description', '')
        if amount > 0:
            result_rows.append(build_row(i, amount, dt, desc))

    if not result_rows:
        return pd.DataFrame()

    return add_engineered_features(pd.DataFrame(result_rows))


def _rows_from_table(table):
    rows = []
    if not table or len(table) < 2:
        return rows

    DATE_KEYS   = ['date','time','on','txn date','transaction date','value date']
    AMOUNT_KEYS = ['amount','amt','debit','credit','paid','received','sum',
                   'withdrawal','deposit','dr','cr','rs','inr']
    DESC_KEYS   = ['description','desc','detail','narration','merchant',
                   'particulars','remarks','name','to','info','ref']

    header = [str(h).lower().strip().replace('\n',' ') if h else '' for h in table[0]]

    date_col   = next((i for i,h in enumerate(header) if any(k in h for k in DATE_KEYS)), None)
    amount_col = next((i for i,h in enumerate(header) if any(k in h for k in AMOUNT_KEYS)), None)
    desc_col   = next((i for i,h in enumerate(header) if any(k in h for k in DESC_KEYS)), None)

    date_pat   = re.compile(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{1,2}\s+\w{3}\s+\d{4}')
    amount_pat = re.compile(r'[\d,]+\.?\d*')

    for row in table[1:]:
        if not row:
            continue
        try:
            raw_date   = str(row[date_col]).strip()   if date_col   is not None and date_col   < len(row) else ''
            raw_amount = str(row[amount_col]).strip()  if amount_col is not None and amount_col < len(row) else ''
            desc       = str(row[desc_col]).strip()    if desc_col   is not None and desc_col   < len(row) else \
                         ' '.join(str(c) for c in row if c and str(c).strip())

            # fallback: scan whole row for date+amount
            if not date_pat.search(raw_date):
                row_str = ' '.join(str(c) for c in row if c)
                dates = date_pat.findall(row_str)
                if dates:
                    raw_date = dates[0]

            if not amount_pat.search(raw_amount):
                row_str = ' '.join(str(c) for c in row if c)
                amounts = re.findall(r'₹?\s*([\d,]+\.?\d*)', row_str)
                if amounts:
                    raw_amount = amounts[0]

            if date_pat.search(raw_date) and parse_amount(raw_amount) > 0:
                rows.append({'raw_date': raw_date, 'raw_amount': raw_amount,
                             'description': desc})
        except:
            continue
    return rows


def _rows_from_text(text: str):
    rows = []
    date_pat   = re.compile(r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{1,2}\s+\w{3}\s+\d{4})')
    amount_pat = re.compile(r'₹\s*([\d,]+\.?\d*)')

    for line in text.split('\n'):
        line = line.strip()
        if len(line) < 8:
            continue
        dates   = date_pat.findall(line)
        amounts = amount_pat.findall(line)
        if dates and amounts:
            for amt in amounts:
                if parse_amount(amt) > 0:
                    rows.append({'raw_date': dates[0],
                                 'raw_amount': amt,
                                 'description': line})
                    break
    return rows