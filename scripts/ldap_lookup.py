import csv
import getpass
import os
import re
import ssl
import sys
from pathlib import Path

from ldap3 import ALL, SUBTREE, Connection, Server, ServerPool, Tls, FIRST
from ldap3.utils.conv import escape_filter_chars

BASE_DIR = Path(__file__).resolve().parent.parent
ATTRS = ["displayName", "mail", "userPrincipalName", "sAMAccountName", "department", "title"]


def load_env():
    env = {}
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    for key in list(env):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def build_pool(env):
    urls = [u for u in re.split(r"[,\s;]+", env.get("LDAP_URLS", "")) if u]
    if not urls:
        sys.exit("LDAP_URLS non configurato nel .env")
    ca_file = env.get("LDAP_CA_FILE")
    if ca_file and not os.path.isabs(ca_file):
        ca_file = str(BASE_DIR / ca_file)
    tls = Tls(validate=ssl.CERT_REQUIRED, ca_certs_file=ca_file) if ca_file else Tls(validate=ssl.CERT_REQUIRED)
    timeout = int(env.get("LDAP_TIMEOUT_SECONDS", "10") or 10)
    servers = [Server(u, use_ssl=u.lower().startswith("ldaps"), tls=tls, get_info=ALL, connect_timeout=timeout) for u in urls]
    return ServerPool(servers, FIRST, active=True, exhaust=True)


def read_names(path):
    names = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.reader(f, delimiter=";" if path.lower().endswith(".csv") else "\t"):
            if row and row[0].strip():
                names.append(row[0].strip())
    return names


def search(conn, base_dn, name):
    value = escape_filter_chars(name)
    flt = f"(&(objectCategory=person)(objectClass=user)(anr={value}))"
    conn.search(base_dn, flt, SUBTREE, attributes=ATTRS, size_limit=10)
    results = []
    for e in conn.entries:
        results.append({a: (str(e[a].value) if a in e and e[a].value else "") for a in ATTRS})
    return results


def main():
    if len(sys.argv) < 2:
        sys.exit("Uso: ./venv/bin/python scripts/ldap_lookup.py nomi.txt [output.csv]")
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "ldap_lookup_result.csv"

    env = load_env()
    base_dn = env.get("LDAP_BASE_DN")
    if not base_dn:
        sys.exit("LDAP_BASE_DN non configurato nel .env")

    names = read_names(input_file)
    if not names:
        sys.exit("Nessun nome trovato nel file di input")

    user = input("Utente AD (UPN, es. nome@dominio): ").strip()
    password = getpass.getpass("Password: ")

    conn = Connection(build_pool(env), user=user, password=password, auto_bind=True, receive_timeout=30)

    found = ambiguous = missing = 0
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["nome_input", "esito", "displayName", "mail", "userPrincipalName", "sAMAccountName", "department", "title"])
        for name in names:
            results = search(conn, base_dn, name)
            if not results:
                missing += 1
                writer.writerow([name, "NON TROVATO", "", "", "", "", "", ""])
                print(f"NON TROVATO  {name}")
                continue
            esito = "OK" if len(results) == 1 else f"AMBIGUO ({len(results)})"
            if len(results) == 1:
                found += 1
            else:
                ambiguous += 1
            for r in results:
                writer.writerow([name, esito] + [r[a] for a in ATTRS])
            print(f"{esito:<12} {name} -> {', '.join(r['mail'] or r['userPrincipalName'] for r in results)}")

    conn.unbind()
    print(f"\nTotale {len(names)} | OK {found} | Ambigui {ambiguous} | Non trovati {missing}")
    print(f"Risultato salvato in {output_file}")


if __name__ == "__main__":
    main()
