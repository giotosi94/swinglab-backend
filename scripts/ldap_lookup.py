import sys
import os
import re
import csv
import getpass
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldap3 import Server, Connection, Tls, ALL, SIMPLE
from ldap3.utils.conv import escape_filter_chars
import ssl

from app.config import settings

HOSTS = ["ITADS103.it.lindt.net", "ITADS104.it.lindt.net"]
ATTRS = ["displayName", "mail", "userPrincipalName", "sAMAccountName", "l", "msDS-cloudExtensionAttribute1", "userAccountControl"]
ENABLED = "(!(userAccountControl:1.2.840.113556.1.4.803:=2))"


def norm_tokens(text):
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower().replace("'", "").replace("\u2019", "").replace("`", "")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return [x for x in t.split() if x]


def ldap_tokens(name):
    out = []
    for raw in name.split():
        if raw.endswith("'") or raw.endswith("\u2019"):
            core = raw[:-1]
            core = core[:-1] if len(core) > 3 else core
            out.append(core)
        elif "'" in raw or "\u2019" in raw:
            parts = re.split(r"['\u2019]", raw)
            out.append(max(parts, key=len))
        else:
            out.append(raw)
    return [t for t in out if len(t) >= 3]


def read_input(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = [l.strip() for l in f.read().splitlines() if l.strip()]
    records = []
    i = 0
    n = len(lines)
    found_triplet = False
    while i < n:
        if (re.fullmatch(r"\d{6,10}", lines[i]) and i + 2 < n
                and not re.fullmatch(r"\d{6,10}", lines[i + 1])
                and re.fullmatch(r"\d{6,10}", lines[i + 2])):
            records.append((lines[i], lines[i + 1], lines[i + 2]))
            found_triplet = True
            i += 3
        else:
            i += 1
    if found_triplet:
        return records
    return [("", l, "") for l in lines if not re.fullmatch(r"\d+", l)]


def connect(upn, password):
    last = None
    for host in HOSTS:
        try:
            tls = Tls(validate=ssl.CERT_REQUIRED, ca_certs_file=settings.LDAP_CA_FILE)
            server = Server(host, port=636, use_ssl=True, tls=tls, get_info=ALL)
            conn = Connection(server, user=upn, password=password, authentication=SIMPLE, auto_bind=True)
            return conn, server
        except Exception as e:
            last = e
    raise RuntimeError("Connessione LDAP fallita: %s" % last)


def get_base(server):
    base = getattr(settings, "LDAP_BASE_DN", None) or getattr(settings, "LDAP_BASE", None)
    if base:
        return base
    return server.info.other["defaultNamingContext"][0]


def run_search(conn, base, tokens, enabled_only=True):
    parts = "".join("(displayName=*%s*)" % escape_filter_chars(t) for t in tokens)
    flt = "(&(objectCategory=person)(objectClass=user)%s%s)" % (ENABLED if enabled_only else "", parts)
    conn.search(base, flt, attributes=ATTRS, size_limit=25)
    rows = []
    for e in conn.entries:
        a = e.entry_attributes_as_dict
        def g(k):
            v = a.get(k)
            if isinstance(v, list):
                return v[0] if v else ""
            return v or ""
        rows.append({
            "display": str(g("displayName")),
            "mail": str(g("mail")),
            "upn": str(g("userPrincipalName")),
            "sam": str(g("sAMAccountName")),
            "sede": str(g("l")),
            "cdc": str(g("msDS-cloudExtensionAttribute1")),
        })
    return rows


def pick(name, rows):
    want = set(norm_tokens(name))
    exact = [r for r in rows if set(norm_tokens(r["display"])) == want]
    if len(exact) == 1:
        return exact[0], "TROVATO", exact
    if len(exact) > 1:
        return None, "AMBIGUO", exact
    long_want = [t for t in want if len(t) >= 3]
    cont = [r for r in rows if all(t in set(norm_tokens(r["display"])) for t in long_want)]
    if len(cont) == 1:
        return cont[0], "APPROSSIMATO", cont
    if len(cont) > 1:
        return None, "AMBIGUO", cont
    return None, "NON_TROVATO", []


def lookup(conn, base, name):
    toks = ldap_tokens(name)
    if not toks:
        return None, "NON_TROVATO", []
    attempts = [toks]
    if len(toks) > 2:
        attempts.append([toks[0], toks[1]])
        attempts.append([toks[0], toks[-1]])
    for t in attempts:
        rows = run_search(conn, base, t)
        if rows:
            row, status, cands = pick(name, rows)
            if status != "NON_TROVATO":
                return row, status, cands
    rows = run_search(conn, base, toks, enabled_only=False)
    if rows:
        row, status, cands = pick(name, rows)
        if status in ("TROVATO", "APPROSSIMATO"):
            return row, "DISABILITATO_" + status, cands
        return row, status, cands
    return None, "NON_TROVATO", []


def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/ldap_lookup.py lista.txt [output.csv]")
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "lista_email.csv"
    records = read_input(src)
    print("Persone lette: %d" % len(records))
    upn = input("UPN per il bind (es. gtosi@lindt.com): ").strip()
    pwd = getpass.getpass("Password: ")
    conn, server = connect(upn, pwd)
    base = get_base(server)
    print("Base DN: %s" % base)
    counts = {}
    out = []
    for idx, (pid, name, mid) in enumerate(records, 1):
        row, status, cands = lookup(conn, base, name)
        counts[status] = counts.get(status, 0) + 1
        r = row or {}
        extra = ""
        if status == "AMBIGUO":
            extra = " | ".join("%s <%s>" % (c["display"], c["upn"]) for c in cands[:6])
        out.append([pid, name, mid, status, r.get("upn", ""), r.get("mail", ""), r.get("display", ""), r.get("sede", ""), r.get("cdc", ""), extra])
        if idx % 25 == 0:
            print("%d/%d" % (idx, len(records)))
    with open(dst, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "nome_lista", "id_manager", "esito", "upn", "mail", "display_name_ad", "sede", "centro_costo", "candidati"])
        w.writerows(out)
    conn.unbind()
    print("Fatto. File: %s" % dst)
    for k in sorted(counts):
        print("%s: %d" % (k, counts[k]))


if __name__ == "__main__":
    main()
