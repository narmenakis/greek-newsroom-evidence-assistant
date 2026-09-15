import csv
import argparse
import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
REQUEST_DELAY = 0.5
REQUEST_TIMEOUT = 20

session = requests.Session()
session.headers.update(HEADERS)
session.mount(
    'https://',
    HTTPAdapter(
        max_retries=Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({'GET'}),
        )
    ),
)
session.mount('http://', session.adapters['https://'])


def fetch(url):
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response


def extract_meta(soup, *keys):
    for key in keys:
        for attr in ['name', 'property']:
            meta = soup.find('meta', attrs={attr: key})
            if meta and meta.get('content'):
                return meta['content'].strip()
    return ''


KATHIMERINI_SECTION_MAP = {
    'society': 'Ελλάδα / Κοινωνία',
    'politics': 'Πολιτική',
    'world': 'Διεθνή',
    'culture': 'Τέχνες / Πολιτισμός',
    'economy': 'Οικονομία',
    'opinions': 'Απόψεις',
    'technology': 'Τεχνολογία',
    'sports': 'Αθλητισμός',
}


def extract_author_kathimerini(soup):
    text = soup.get_text()
    m = re.search(r'(?:Newsroom|Ιωάννα Μάνδρου)', text)
    if m:
        return m.group(0)
    return 'Newsroom'


def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def scrape_kathimerini(url):
    r = fetch(url)
    soup = BeautifulSoup(r.content, 'lxml')

    title_tag = soup.find('h1')
    title = clean_text(title_tag.get_text()) if title_tag else ''

    body_el = soup.select_one('.entry-content')
    body_text = ''
    if body_el:
        paras = body_el.find_all('p')
        body_text = ' '.join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
        body_text = clean_text(body_text)

    pub_date = extract_meta(soup, 'article:published_time')
    if pub_date:
        pub_date = pub_date[:10]

    section = ''
    path_parts = urlparse(url).path.strip('/').split('/')
    if path_parts:
        raw = path_parts[0].lower()
        section = KATHIMERINI_SECTION_MAP.get(raw, raw.replace('-', ' / ').title())

    author = extract_author_kathimerini(soup)
    website = 'kathimerini.gr'

    return url, title, body_text, author, website, pub_date, section


def scrape_efsyn(url):
    r = fetch(url)
    soup = BeautifulSoup(r.content, 'lxml')

    title_tag = soup.find('h1')
    title = clean_text(title_tag.get_text()) if title_tag else ''

    body_el = soup.select_one('.article-body')
    body_text = ''
    if body_el:
        paras = body_el.find_all('p')
        body_text = ' '.join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
        body_text = clean_text(body_text)

    pub_date = extract_meta(soup, 'article:published_time')
    if pub_date:
        pub_date = pub_date[:10]

    section = extract_meta(soup, 'article:section')

    author_el = soup.select_one('.field-name-field-author .field-item, .author, .byline')
    author = clean_text(author_el.get_text()) if author_el else 'Newsroom'
    if not author or author.lower() in ('efsyn.gr', ''):
        author = 'Newsroom'
    website = 'efsyn.gr'

    return url, title, body_text, author, website, pub_date, section


def scrape_skai(url):
    r = fetch(url)
    soup = BeautifulSoup(r.content, 'lxml')

    title_tag = soup.find('h1')
    title = clean_text(title_tag.get_text()) if title_tag else ''

    body_el = soup.select_one('.post-content')
    body_text = ''
    if body_el:
        paras = body_el.find_all('p')
        body_text = ' '.join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
        body_text = clean_text(body_text)

    pub_date = extract_meta(soup, 'article:published_time')
    if pub_date:
        pub_date = pub_date[:10]

    section = extract_meta(soup, 'article:section')
    author = extract_meta(soup, 'article:author')
    if not author:
        author = 'Newsroom'
    website = 'skai.gr'

    return url, title, body_text, author, website, pub_date, section


def scrape_zougla(url):
    r = fetch(url)
    soup = BeautifulSoup(r.content, 'lxml')

    title_tag = soup.find('h1')
    title = clean_text(title_tag.get_text()) if title_tag else ''

    body_el = soup.select_one('.entry-content')
    body_text = ''
    if body_el:
        paras = body_el.find_all('p')
        body_text = ' '.join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
        body_text = clean_text(body_text)

    pub_date = extract_meta(soup, 'article:published_time')
    if pub_date:
        pub_date = pub_date[:10]

    section = extract_meta(soup, 'article:section')

    text = soup.get_text()
    author_match = re.search(r'([Α-ΩΆ-Ώ][α-ωά-ώ]+\s[Α-ΩΆ-Ώ][α-ωά-ώ]+)\s+\d{2}\.\d{2}\.\d{4}', text)
    author = clean_text(author_match.group(1)) if author_match else 'Newsroom'
    website = 'zougla.gr'

    return url, title, body_text, author, website, pub_date, section


SCRAPERS = {
    'kathimerini.gr': scrape_kathimerini,
    'efsyn.gr': scrape_efsyn,
    'skai.gr': scrape_skai,
    'zougla.gr': scrape_zougla,
}


def get_scraper(url):
    domain = urlparse(url).netloc.lower()
    for key in SCRAPERS:
        if key in domain:
            return SCRAPERS[key]
    return None


def is_not_found_result(result):
    """Reject common CMS error pages instead of indexing them as articles."""
    title, body = result[1], result[2]
    return bool(re.search(r'(?:σελίδα\s+δεν\s+βρέθηκε|page\s+not\s+found|404\s+not\s+found)', title, re.I)) or not body.strip()


def write_rows(path, rows):
    """Write a complete CSV snapshot so an interrupted run is recoverable."""
    temporary = f'{path}.tmp'
    with open(temporary, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'title', 'full-text', 'author', 'website', 'datetime', 'section'])
        writer.writerows(rows)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description='Scrape the configured article URL list')
    parser.add_argument('--input', default='links.txt')
    parser.add_argument('--output', default='sample_database.csv')
    parser.add_argument('--resume', action='store_true', help='Resume from the output checkpoint')
    parser.add_argument('--delay', type=float, default=REQUEST_DELAY)
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    checkpoint_file = f'{output_file}.checkpoint.json'

    urls = []
    with open(input_file, 'r', encoding='utf-8') as f:
        urls = list(dict.fromkeys(line.strip() for line in f if line.strip()))

    print(f"Loaded {len(urls)} URLs from {input_file}")

    rows = []
    completed = set()
    if args.resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        rows = checkpoint.get('rows', [])
        completed = set(checkpoint.get('completed_urls', []))
        print(f"Resuming: {len(completed)} URLs already completed")

    total = len(urls)
    for i, url in enumerate(urls, 1):
        if url in completed:
            continue
        scraper = get_scraper(url)
        if not scraper:
            print(f"[{i}/{total}] ⚠️ No scraper for {url}")
            completed.add(url)
            continue

        try:
            result = scraper(url)
            if is_not_found_result(result):
                print(f"[{i}/{total}] ⚠️ Rejected not-found/empty page: {url}")
            else:
                rows.append(result)
                print(f"[{i}/{total}] ✅ {urlparse(url).netloc} - {result[1][:50]}")
            completed.add(url)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 404:
                # A permanent missing page should not be retried forever when
                # the user resumes an otherwise complete scrape.
                completed.add(url)
                print(f"[{i}/{total}] ⚠️ Rejected HTTP 404: {url}")
            else:
                print(f"[{i}/{total}] ❌ {url}: {e}")
        except Exception as e:
            print(f"[{i}/{total}] ❌ {url}: {e}")

        write_rows(output_file, rows)
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({'completed_urls': sorted(completed), 'rows': rows}, f, ensure_ascii=False)
        time.sleep(max(args.delay, 0))

    print(f"\nSaved {len(rows)} articles to {output_file}")


if __name__ == '__main__':
    main()
