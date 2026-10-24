#!/usr/bin/env python3
"""Fetch Crossref metadata for candidate DOIs and print elsarticle bibitems."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

MAIL = "jjp@mails.guet.edu.cn"
UA = f"info-consistent-rul/1.0 (mailto:{MAIL})"

DOIS = [
    "10.1038/s41560-019-0356-8",
    "10.1016/j.joule.2023.07.014",
    "10.1145/2382577.2382579",
    "10.1016/j.patter.2023.100804",
    "10.1007/s10618-022-00894-5",
    "10.1016/j.jpowsour.2011.08.040",
    "10.1038/s41586-020-1994-5",
    "10.1038/s42256-021-00312-3",
    "10.1016/j.rser.2015.11.040",
    "10.1016/j.apenergy.2017.07.001",
    "10.1016/j.jclepro.2018.06.016",
    "10.1016/j.ejor.2010.11.018",
    "10.1016/j.ymssp.2017.11.016",
    "10.1016/j.jpowsour.2013.01.094",
    "10.1016/j.jpowsour.2017.05.053",
    "10.1016/j.jpowsour.2017.05.004",
    "10.1109/TR.2008.2013682",
    "10.1016/j.microrel.2012.12.004",
    "10.1016/j.microrel.2012.12.011",
    "10.1016/j.jpowsour.2013.12.011",
    "10.1016/j.ress.2016.11.013",
    "10.1016/j.joule.2021.06.005",
    "10.1038/s41560-020-00697-2",
    "10.1016/j.jpowsour.2012.02.084",
    "10.1038/s41467-020-19776-9",
    "10.1109/TIE.2020.2973876",
    "10.1016/j.energy.2018.06.053",
    "10.1016/j.jpowsour.2017.11.077",
    "10.1016/j.energy.2019.116277",
    "10.1016/j.energy.2019.01.009",
    "10.1016/j.egyai.2021.100081",
    "10.1016/j.rser.2014.07.157",
    "10.1109/TAES.2008.4575220",
    "10.1016/j.jpowsour.2014.01.118",
    "10.1016/j.ensm.2021.10.019",
    "10.1016/j.joule.2020.06.006",
    "10.1038/s41560-021-00863-0",
    "10.1016/j.jpowsour.2013.02.029",
]


def fetch(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))["message"]


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
        p = msg.get(k, {}).get("date-parts")
        if p:
            return str(p[0][0])
    return ""


def pages(msg: dict) -> str:
    return msg.get("page") or msg.get("article-number") or ""


def short_j(msg: dict) -> str:
    t = (msg.get("short-container-title") or msg.get("container-title") or [""])[0]
    return t


def main() -> None:
    rows = []
    for doi in DOIS:
        try:
            m = fetch(doi)
            rec = {
                "doi": doi,
                "title": (m.get("title") or [""])[0],
                "authors": authors(m),
                "year": year(m),
                "journal": short_j(m),
                "volume": m.get("volume") or "",
                "issue": m.get("issue") or "",
                "pages": pages(m),
            }
            rows.append(rec)
            print(
                f"{rec['authors']},\n{rec['title']},\n{rec['journal']} {rec['volume']}"
                f"{(' ('+rec['issue']+')') if rec['issue'] else ''} ({rec['year']}) {rec['pages']}.\n"
                f"doi:{doi}\n"
            )
        except Exception as e:
            print("FAIL", doi, e)
        time.sleep(0.12)
    out = Path(__file__).resolve().parents[1] / "outputs" / "info_levels_v2" / "ref_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out, "n=", len(rows))


if __name__ == "__main__":
    main()
