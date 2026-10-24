#!/usr/bin/env python3
from __future__ import annotations
import json, time, urllib.parse, urllib.request
from pathlib import Path

MAIL = "jjp@mails.guet.edu.cn"
UA = f"info-consistent-rul/1.0 (mailto:{MAIL})"

DOIS = [
    "10.1016/j.joule.2023.07.021",
    "10.1016/j.rser.2015.12.009",
    "10.1016/j.apenergy.2017.12.016",
    "10.1016/j.jclepro.2018.08.164",
    "10.1016/j.jpowsour.2013.12.110",
    "10.1016/j.rser.2019.109334",
    "10.1016/j.jpowsour.2012.11.126",
    "10.1016/j.microrel.2012.12.003",
    "10.1109/TAES.2008.4656504",
    "10.1109/TR.2008.2013673",
    "10.1016/j.jpowsour.2012.07.054",
    "10.1016/j.jpowsour.2016.12.011",
    "10.1016/j.jpowsour.2017.11.040",
    "10.1016/j.energy.2018.04.119",
    "10.1016/j.energy.2018.06.171",
    "10.1016/j.energy.2019.05.161",
    "10.1016/j.jpowsour.2018.11.081",
    "10.1038/s42256-020-00228-4",
    "10.1093/imamat/hxw041",
    "10.5555/3295222.3295349",
    "10.48550/arXiv.2211.14730",
    "10.1016/j.jpowsour.2022.231127",
    "10.1038/s41560-020-00747-9",
    "10.1016/j.joule.2020.11.018",
    "10.1016/j.egyai.2022.100158",
    "10.1016/j.jpowsour.2014.09.146",
    "10.1016/j.apenergy.2019.113817",
    "10.1016/j.ress.2017.05.019",
    "10.1016/j.jpowsour.2011.08.040",
    "10.1016/j.patter.2022.100444",
    "10.1038/s41467-024-46150-w",
    "10.1109/ACCESS.2019.2940727",
    "10.1016/j.rser.2021.111480",
    "10.1016/j.apenergy.2020.115406",
    "10.1016/j.est.2021.102585",
    "10.1016/j.jpowsour.2020.228805",
    "10.1016/j.energy.2020.117789",
    "10.1016/j.ymssp.2019.106426",
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
        p = (msg.get(k) or {}).get("date-parts")
        if p:
            return str(p[0][0])
    return ""


def main():
    rows = []
    for doi in DOIS:
        try:
            m = fetch(doi)
            rec = {
                "doi": doi,
                "title": (m.get("title") or [""])[0],
                "authors": authors(m),
                "year": year(m),
                "journal": (m.get("short-container-title") or m.get("container-title") or [""])[0],
                "volume": m.get("volume") or "",
                "issue": m.get("issue") or "",
                "pages": m.get("page") or m.get("article-number") or "",
            }
            rows.append(rec)
            print("OK", rec["year"], rec["journal"], rec["title"][:70], doi)
        except Exception as e:
            print("FAIL", doi, type(e).__name__)
        time.sleep(0.15)
    Path(r"f:\研究生项目\tcn_attention\outputs\info_levels_v2\ref_batch2.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("n", len(rows))


if __name__ == "__main__":
    main()
