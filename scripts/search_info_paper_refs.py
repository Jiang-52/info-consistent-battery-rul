#!/usr/bin/env python3
"""Query Crossref by title and dump verified metadata."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

MAIL = "jjp@mails.guet.edu.cn"
UA = f"info-consistent-rul/1.0 (mailto:{MAIL})"

QUERIES = [
    "Selecting the appropriate features in battery lifetime predictions Geslin",
    "Critical review of the methods for monitoring of lithium-ion batteries in electric vehicles Waag",
    "Review of the remaining useful life prognostics of vehicle lithium-ion batteries Hu",
    "State of charge estimation of lithium-ion batteries Xiong Applied Energy 2018",
    "A review of lithium-ion battery state of health estimation and remaining useful life prediction Lipu",
    "Health diagnosis and remaining useful life prognostics of lithium-ion batteries Rezvanizaniani",
    "Support vector machine for remaining useful life Nuhic lithium",
    "An ensemble model for predicting remaining useful life Xing lithium",
    "Prognostics methods applied to electric vehicles battery Goebel",
    "Comparison of prognostic algorithms for estimating remaining useful life Saha Goebel",
    "State of health diagnosis of lithium-ion batteries Dubarry",
    "Degradation diagnostics for lithium ion cells Birkl",
    "State-of-charge estimation of Li-ion batteries using deep neural networks Chemali",
    "Remaining useful life prediction of lithium-ion batteries using LSTM Zhang Energy",
    "Remaining useful life prediction for lithium-ion battery using convolutional neural network Ren",
    "A novel remaining useful life prediction method for lithium-ion battery based on long short-term memory Li Energy",
    "Remaining useful life prediction of lithium-ion batteries using temporal convolutional network Ma",
    "Predicting the state of charge and health of batteries using data-driven machine learning Ng",
    "Dropout as a Bayesian approximation Gal",
    "Attention is all you need Vaswani",
    "A Time Series is Worth 64 Words PatchTST Nie",
    "Towards reproducible state-of-the-art energy storage performance Attia Nature Energy",
    "Principles of the Battery Data Genome Ward",
    "Identifying degradation mechanisms in lithium-ion batteries Reniers",
    "Gaussian process regression for in situ capacity estimation of lithium-ion batteries in electric vehicles",
    "Battery health prediction under generalized conditions using a Gaussian process prior Greenbank Howey",
    "Closed-loop optimization of fast-charging protocols Attia",
    "Machine learning pipeline for battery state-of-health estimation Roman",
    "The challenge and opportunity of battery lifetime prediction from field data Sulzer",
    "Lithium-ion battery data and where to find it dos Reis",
    "A Data-Driven Approach With Uncertainty Quantification Liu remaining useful life",
    "Prognostics of lithium-ion batteries based on Dempster He Williard",
    "Data-driven prediction of battery cycle life before capacity degradation Severson",
    "Leakage in data mining Kaufman",
    "Leakage and the reproducibility crisis Kapoor",
    "Forecast evaluation for data scientists Hewamalage",
    "Remaining useful life estimation A review statistical data driven Si",
    "Machinery health prognostics systematic review Lei",
    "Remaining useful life prediction of lithium-ion battery unscented particle filter Miao",
    "Gaussian process regression for forecasting battery state of health Richardson",
]


def search(q: str) -> dict | None:
    params = urllib.parse.urlencode(
        {
            "query.bibliographic": q,
            "rows": "3",
            "select": "DOI,title,author,container-title,short-container-title,volume,issue,page,issued,published-print,published-online,type,abstract",
        }
    )
    url = "https://api.crossref.org/works?" + params
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        items = json.loads(r.read().decode("utf-8"))["message"]["items"]
    return items[0] if items else None


def authors(msg: dict) -> str:
    out = []
    for a in msg.get("author") or []:
        fam = a.get("family", "")
        given = a.get("given", "")
        if given:
            initials = ".".join(p[0] for p in given.replace("-", " ").split() if p) + "."
            out.append(f"{initials}~{fam}")
        else:
            out.append(fam)
    return ", ".join(out)


def year(msg: dict) -> str:
    for k in ("published-print", "published-online", "issued"):
        p = (msg.get(k) or {}).get("date-parts")
        if p:
            return str(p[0][0])
    return ""


def main() -> None:
    rows = []
    for q in QUERIES:
        try:
            m = search(q)
            if not m:
                print("NONE", q)
                continue
            rec = {
                "query": q,
                "doi": m.get("DOI"),
                "title": (m.get("title") or [""])[0],
                "authors": authors(m),
                "year": year(m),
                "journal": (m.get("short-container-title") or m.get("container-title") or [""])[0],
                "volume": m.get("volume") or "",
                "issue": m.get("issue") or "",
                "pages": m.get("page") or "",
            }
            rows.append(rec)
            t = rec["title"].replace("\n", " ")[:90]
            print(f"OK {rec['year']} | {rec['journal']} | {t} | {rec['doi']}")
        except Exception as e:
            print("FAIL", q, type(e).__name__, e)
        time.sleep(0.25)
    out = Path(r"f:\研究生项目\tcn_attention\outputs\info_levels_v2\ref_crossref_search.json")
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("n=", len(rows), "->", out)


if __name__ == "__main__":
    main()
