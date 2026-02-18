import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import re

# =========================
# CONFIG
# =========================
aziende_cik = {
    "Microsoft": "0000789019",
    "Apple":     "0000320193",
    "Nvidia":    "0001045810",
}

# Quanti 8-K per azienda vuoi SCANSIONARE (non tutti diventano press release)
N_8K_PER_COMPANY = 300

# Rate limiting (SEC)
SLEEP_SEC = 0.25

headers = {"User-Agent": "TuoNome tuaemail@dominio.com"}  # <-- IMPORTANTE

session = requests.Session()
session.headers.update(headers)

PRESS_KEYWORDS = [
    "press release", "earnings release", "news release",
    "conference call", "prepared remarks", "investor presentation"
]

def sleep():
    time.sleep(SLEEP_SEC)

def get_with_retry(url, tries=4, timeout=60):
    last = None
    for i in range(tries):
        sleep()
        r = session.get(url, timeout=timeout)
        last = r
        if r.status_code == 429:
            time.sleep(1.0 + i * 1.5)
            continue
        if r.status_code >= 500:
            time.sleep(1.0 + i * 1.5)
            continue
        return r
    return last

def get_json(url):
    r = get_with_retry(url)
    r.raise_for_status()
    return r.json()

def clean_html_to_text(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    # rimuovi inline XBRL se presente (tag che iniziano con ix:)
    for ix_tag in soup.find_all(lambda t: t.name and str(t.name).lower().startswith("ix:")):
        ix_tag.decompose()

    # rimuovi elementi nascosti
    for hidden_tag in soup.find_all(style=lambda v: v and "display:none" in v.replace(" ", "").lower()):
        hidden_tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_text_file(name: str) -> bool:
    n = (name or "").lower()
    return n.endswith(".htm") or n.endswith(".html") or n.endswith(".txt")

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def score_item(item: dict) -> int:
    """
    Sceglie l'exhibit migliore per 'press release'
    item tipico (index.json): {"name": "...", "type": "...", "size": ...}
    """
    name = normalize(item.get("name", ""))
    typ  = normalize(item.get("type", ""))

    score = 0

    # Segnale forte: EX-99.1 / 99.01
    if "99.1" in typ or "99.01" in typ or "ex-99" in typ:
        score += 100
    if "ex-99.1" in typ or "ex-99.01" in typ:
        score += 120

    # Anche dal filename
    if "ex99" in name or "ex-99" in name:
        score += 50
    if "99_1" in name or "99-1" in name or "99.1" in name or "9901" in name:
        score += 30

    # Keyword nel filename
    for kw in PRESS_KEYWORDS:
        if kw in name:
            score += 20

    # Solo testo
    if is_text_file(item.get("name", "")):
        score += 10
    else:
        score -= 100  # scarta pdf/jpg/etc

    return score

def list_accession_files(cik: str, accession: str):
    cik_short = str(int(cik))
    accession_clean = accession.replace("-", "")
    idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik_short}/{accession_clean}/index.json"
    try:
        data = get_json(idx_url)
        items = data.get("directory", {}).get("item", [])
        return items, cik_short, accession_clean
    except Exception:
        return [], cik_short, accession_clean

def download_best_press_release(cik: str, accession: str):
    """
    Ritorna dict con testo + metadata, oppure None se non trova exhibit valido.
    """
    items, cik_short, accession_clean = list_accession_files(cik, accession)
    if not items:
        return None

    # filtra solo file testuali candidati
    candidates = [it for it in items if is_text_file(it.get("name", ""))]
    if not candidates:
        return None

    # scegli best
    best = max(candidates, key=score_item)
    if score_item(best) < 20:
        # troppo debole -> probabilmente non è press release
        return None

    filename = best.get("name")
    filetype = best.get("type", "")

    file_url = f"https://www.sec.gov/Archives/edgar/data/{cik_short}/{accession_clean}/{filename}"
    r = get_with_retry(file_url)
    if r.status_code != 200:
        return None

    text = clean_html_to_text(r.content)
    if len(text) < 1000:  # euristica: troppo corto = probabilmente non è la release
        return None

    return {
        "ExhibitFile": filename,
        "ExhibitType": filetype,
        "URL": file_url,
        "Text": text
    }

# =========================
# MAIN
# =========================
rows = []

for azienda, cik in aziende_cik.items():
    print(f"\n[{azienda}] Scarico lista 8-K (target scan: {N_8K_PER_COMPANY})...")

    # prendi filings "recent" (e basta) -> semplice e veloce; per storia lunga dovresti aggiungere i blocks
    sub_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    data = get_json(sub_url)
    recent = data.get("filings", {}).get("recent", {})

    df = pd.DataFrame({
        "Data": recent.get("filingDate", []),
        "Ora": recent.get("acceptanceDateTime", []),
        "Tipo": recent.get("form", []),
        "Accession": recent.get("accessionNumber", []),
    })

    df_8k = df[df["Tipo"] == "8-K"].sort_values(["Data", "Ora"], ascending=False).head(N_8K_PER_COMPANY).copy()
    print(f"[{azienda}] 8-K da scandire: {len(df_8k)}")

    found = 0
    for _, r in df_8k.iterrows():
        accession = r["Accession"]
        filing_date = r["Data"]
        filing_time = r["Ora"]

        pr = download_best_press_release(cik, accession)
        if pr is None:
            continue

        rows.append({
            "Azienda": azienda,
            "CIK": cik,
            "FilingDate": filing_date,
            "AcceptanceDateTime": filing_time,
            "Accession": accession,
            "ExhibitType": pr["ExhibitType"],
            "ExhibitFile": pr["ExhibitFile"],
            "URL": pr["URL"],
            "Text": pr["Text"],
        })
        found += 1

        if found % 10 == 0:
            print(f"[{azienda}] Press releases trovate: {found}")

    print(f"[{azienda}] Totale press releases trovate: {found}")

df_out = pd.DataFrame(rows)
df_out.to_csv("SEC_8K_PressReleases.csv", index=False, encoding="utf-8")
print(f"\nSALVATO: SEC_8K_PressReleases.csv | righe: {len(df_out)}")
print(df_out.head(3))
