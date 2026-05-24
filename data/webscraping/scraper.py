'''
MediExplain SG — Knowledge Base Scraper

MediExplain SG lets patients upload their discharge summaries, lab reports,
and insurance claims and get plain-language explanations back.

To do that, the system needs a knowledge base — a collection of medical
reference articles it can retrieve from when constructing an explanation.
This script builds that knowledge base by scraping two sections of HealthHub
(Singapore's national health portal, run by MOH):
  - /health-conditions/                → conditions patients are diagnosed with
  - /medication-devices-and-treatment/ → drugs and devices on discharge summaries

Output: one JSON file per page saved to data/webscraping/raw/.
Each file includes a "category" field so the ingest service can tag
embeddings by type when loading them into pgvector.

Next step: the ingest service reads these files, chunks the text,
generates embeddings, and loads them into pgvector for RAG retrieval.
'''

import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import structlog
from scrapling.fetchers import Fetcher

# ---------------------------------------------------------------------------
# Logging
# structlog emits one JSON object per log line.
# Each line has: timestamp, level, event, plus any extra key=value pairs.
# JSON format means you can pipe output through `jq` to filter and search.
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Test mode
# When TEST_MODE is True, the scraper only runs on TEST_PAGES instead of
# fetching the full sitemap. Flip to False once output looks correct.
# ---------------------------------------------------------------------------
TEST_MODE = False

TEST_PAGES = [
    # Known clean article
    {"url": "https://www.healthhub.sg/health-conditions/diabetes",
     "slug": "diabetes", "category": "health-condition"},
    # Known noisy article (has nav/footer mixed in)
    {"url": "https://www.healthhub.sg/health-conditions/bruises_hpb",
     "slug": "bruises_hpb", "category": "health-condition"},
    # Known JS-rendered page (content was empty before)
    {"url": "https://www.healthhub.sg/health-conditions/topic_dengue_fever_moh",
     "slug": "topic_dengue_fever_moh", "category": "health-condition"},
    # Medication page
    {"url": "https://www.healthhub.sg/medication-devices-and-treatment/medications/metformin",
     "slug": "metformin", "category": "medication-devices-treatment"},
    # Handling medications page
    {"url": "https://www.healthhub.sg/medication-devices-and-treatment/handling-medications/storage-and-expiry",
     "slug": "storage-and-expiry", "category": "medication-devices-treatment"},
    # Previously hollow JS-rendered pages
    {"url": "https://www.healthhub.sg/health-conditions/kidney-stones",
     "slug": "kidney-stones", "category": "health-condition"},
    {"url": "https://www.healthhub.sg/health-conditions/abscess-dental",
     "slug": "abscess-dental", "category": "health-condition"},
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SITEMAP_URL = "https://www.healthhub.sg/sitemap.xml"

CATEGORIES = {
    "/health-conditions/": "health-condition",
    "/medication-devices-and-treatment/": "medication-devices-treatment",
}

OUTPUT_DIR = Path(__file__).parent / "raw"

# Seconds to wait between requests.
RATE_LIMIT = 0.5

# Paragraphs shorter than this are almost always single navigation labels
# (e.g. "Diabetes", "Cancer", "R", "I", "C", "E" from the RICE acronym headers).
# 25 characters filters those while keeping real short sentences like
# "Rest the bruised area if possible" (33 chars).
MIN_PARAGRAPH_LENGTH = 25

# Strings that appear in HealthHub's navigation and footer on every page.
# Any paragraph containing one of these is dropped.
NOISE_PATTERNS = [
    # Top navigation bar — appears on every page
    "handling medications",
    # Footer / newsletter
    "stay up to date with the latest news",
    "join our newsletter mailing list",
    "follow us:",
    "share article",
    "by subscribing to our e-newsletter",
    "and acknowledged that you have read and understood our",
    "this site is protected by recaptcha",
    "and\n\napply",
    # Recurring sidebar / related content labels
    "ministry of health singapore",
    "healthier sg screening",
    "find recommended health screening",
    "explore some of these related topics",
    "medication information leaflet",
    "q&a: wrist and chronic back pain",
    "q&a: skin bumps",
    "q&a: spicy food",
    "q&a: expired health supplements",
    "recommended dietary allowances",
    "how many calories do i need",
    "5 exercises to prevent chronic",
    "all you need to know about childhood immunisations",
    "blood glucose meter to monitor",
    "women health services",
    "community health assist scheme",
    "vaccination clinic",
]

# If the body after filtering is shorter than this, the page has no real
# content (likely a JS-rendered page our fetcher can't see).
MIN_BODY_LENGTH = 100


# ---------------------------------------------------------------------------
# Noise filtering
# ---------------------------------------------------------------------------

def is_noise(paragraph: str) -> bool:
    '''
    Return True if a paragraph should be discarded.
    A paragraph is noise if it is too short or matches a known
    navigation/footer pattern.
    '''
    text = paragraph.strip()

    if len(text) < MIN_PARAGRAPH_LENGTH:
        return True

    text_lower = text.lower()
    for pattern in NOISE_PATTERNS:
        if pattern in text_lower:
            return True

    return False


# ---------------------------------------------------------------------------
# Sitemap discovery
# ---------------------------------------------------------------------------

def get_pages_from_sitemap() -> list[dict]:
    '''
    Fetch the HealthHub sitemap, parse the XML, and return every page
    that matches one of the patterns in CATEGORIES.

    Returns a list of dicts, each with:
      - url:      the full URL of the page
      - slug:     the last path segment (e.g. "diabetes")
      - category: what type of content it is (e.g. "health-condition")
    '''
    log.info("fetching_sitemap", url=SITEMAP_URL)

    with urllib.request.urlopen(SITEMAP_URL) as response:
        xml_bytes = response.read()

    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ET.fromstring(xml_bytes)

    pages = []
    for url_el in root.findall("sm:url", namespace):
        loc = url_el.findtext("sm:loc", namespaces=namespace) or ""

        for pattern, category in CATEGORIES.items():
            if pattern in loc:
                slug = loc.rstrip("/").split("/")[-1]
                pages.append({"url": loc, "slug": slug, "category": category})
                break

    counts = {cat: sum(1 for p in pages if p["category"] == cat) for cat in CATEGORIES.values()}
    log.info("sitemap_parsed", total=len(pages), by_category=counts)
    return pages


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def scrape_page(url: str, slug: str, category: str) -> dict | None:
    '''Fetch one page, filter noise from the body, and return a clean dict.'''
    log.info("scraping", slug=slug, category=category, url=url)

    try:
        page = Fetcher().get(url)

        title = page.css("h1::text").get()

        scrollable = page.css(".in-page-scrollable *::text").getall()
        raw_paragraphs = scrollable if scrollable else page.css("p::text").getall()

        # Apply noise filter — drop short lines and known nav/footer patterns.
        clean_paragraphs = [p.strip() for p in raw_paragraphs if not is_noise(p)]
        body = "\n\n".join(clean_paragraphs)

        if not title:
            log.warning("no_title", slug=slug)
            return None

        if len(body) < MIN_BODY_LENGTH:
            log.warning("empty_after_filter", slug=slug, body_len=len(body))
            return None

        return {
            "slug": slug,
            "url": url,
            "category": category,
            "title": title,
            "body": body,
            "source": "healthhub.sg",
        }

    except Exception as e:
        log.error("scrape_failed", slug=slug, url=url, error=str(e))
        return None


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_article(data: dict) -> None:
    '''Write article data to a JSON file in OUTPUT_DIR.'''
    filename = f"{data['category']}-{data['slug']}.json"
    path = OUTPUT_DIR / filename

    if path.exists():
        log.info("already_exists", slug=data["slug"], path=str(path))
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log.info("saved", slug=data["slug"], category=data["category"], path=str(path))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if TEST_MODE:
        pages = TEST_PAGES
        log.info("test_mode", total=len(pages))
    else:
        pages = get_pages_from_sitemap()

    total = len(pages)
    log.info("scraper_started", total=total, output_dir=str(OUTPUT_DIR))

    saved = 0
    failed = 0

    for i, page in enumerate(pages, start=1):
        log.info("progress", current=i, total=total)

        data = scrape_page(
            url=page["url"],
            slug=page["slug"],
            category=page["category"],
        )

        if data:
            save_article(data)
            saved += 1
        else:
            failed += 1

        if i < total:
            time.sleep(RATE_LIMIT)

    log.info("scraper_finished", saved=saved, failed=failed, total=total)


if __name__ == "__main__":
    main()
