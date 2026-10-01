import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

import tldextract

ROOT = Path(__file__).resolve().parents[2]
WHITELIST_PATH = ROOT / "data" / "raw" / "pk_official_institutions.csv"

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?\S*)", re.IGNORECASE)

_whitelist = {}


def _load_whitelist():
    if not _whitelist:
        df = pd.read_csv(WHITELIST_PATH)
        for _, row in df.iterrows():
            name = row["institution_name"].lower()
            domain = row["official_domain"].lower()
            _whitelist[name] = domain
    return _whitelist


def extract_urls(text):
    return URL_PATTERN.findall(text)


def resolve_url(url):
    if not url.startswith("http"):
        url = "http://" + url
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return urlparse(response.url).netloc.lower()
    except requests.RequestException:
        return None


def extract_domain(url):
    if not url.startswith("http"):
        url = "http://" + url
    return urlparse(url).netloc.lower()


def registered_domain(url):
    if not url.startswith("http"):
        url = "http://" + url
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}".lower()


def find_claimed_brand(text, brand_list):
    text_lower = text.lower()
    for brand in brand_list:
        if brand.lower() in text_lower:
            return brand
    return None


def check_url_mismatch(text, brand_list):
    whitelist = _load_whitelist()
    urls = extract_urls(text)
    claimed_brand = find_claimed_brand(text, brand_list)

    if not urls or not claimed_brand:
        return {"checked": False, "reason": "no URL or no claimed brand found"}

    official_domain = None
    for name, domain in whitelist.items():
        if claimed_brand.lower() in name:
            official_domain = domain
            break

    if official_domain is None:
        return {"checked": False, "reason": f"'{claimed_brand}' not found in whitelist"}

    resolved = resolve_url(urls[0])
    raw_actual = resolved if resolved else urls[0]
    actual_domain = registered_domain(raw_actual)
    official_registered = registered_domain(official_domain)

    mismatch = actual_domain != official_registered
    return {
        "checked": True,
        "claimed_brand": claimed_brand,
        "official_domain": official_domain,
        "actual_domain": actual_domain,
        "mismatch": mismatch,
    }
