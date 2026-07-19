"""
generate_dataset.py
--------------------
Builds a labeled dataset of REAL and FAKE news articles for the Fake News
Detector project.

Why a generated dataset?
Public fake-news corpora (e.g. Kaggle's Fake and Real News Dataset,
LIAR, FakeNewsNet) are the standard choice for this exact project and are
recommended in the README so you can swap in a bigger dataset later.
Since this environment does not have internet access to download them,
this script programmatically builds a large, linguistically varied
dataset that captures the SAME stylistic signals researchers use to
distinguish real vs fake news:

REAL news style           -> attributed, specific, measured tone
  - named sources ("according to", "officials said", "the report found")
  - concrete numbers, dates, organizations
  - neutral/formal vocabulary, few exclamation marks

FAKE news style           -> sensational, unattributed, clickbait tone
  - ALL-CAPS / exclamation-heavy phrasing
  - absolute claims ("SHOCKING", "you won't believe", "secret", "banned")
  - vague or missing sourcing ("sources say", "anonymous insider")
  - conspiratorial / emotionally-loaded language

Combining topic x template x entity substitution produces thousands of
distinct sentences, so the model learns genuine stylistic patterns
instead of memorizing a handful of examples.

Output: data/news_dataset.csv  (columns: text, label)   label: 1 = REAL, 0 = FAKE
"""

import csv
import random
import os

random.seed(42)

TOPICS = ["politics", "health", "technology", "science", "business",
          "sports", "entertainment", "world", "environment", "education"]

PEOPLE = ["the Prime Minister", "the Health Minister", "Dr. Aisha Rahman",
          "Senator Mark Devlin", "CEO Lena Ortiz", "Professor Wei Zhang",
          "Governor Sam Whitfield", "the Finance Secretary", "coach Ravi Menon",
          "the mayor", "the Education Minister", "researcher Elena Petrova"]

ORGS = ["the World Health Organization", "NASA", "the Reserve Bank",
        "Harvard University", "the United Nations", "Reuters",
        "the Ministry of Health", "Google", "the National Weather Service",
        "the Election Commission", "the World Bank", "Interpol"]

PLACES = ["New Delhi", "Lucknow", "Washington", "London", "Geneva",
          "Mumbai", "Tokyo", "Berlin", "Nairobi", "Sydney", "Toronto", "Paris"]

NUMBERS = ["3.2%", "12,000", "45", "1.8 million", "60%", "7", "2.5 billion",
           "18 months", "200", "9.4%", "3,400", "72 hours"]

YEARS = ["2023", "2024", "2025", "2026"]

# ---------------------------------------------------------------------------
# REAL NEWS TEMPLATES  (attributed, specific, neutral tone)
# ---------------------------------------------------------------------------
REAL_TEMPLATES = [
    "{org} announced on {day} that {topic} spending will rise by {num}, according to a statement released in {place}.",
    "{person} said the new {topic} policy would be reviewed within {num} days, according to officials in {place}.",
    "A report published by {org} found that {topic} indicators improved by {num} during {year}, researchers said.",
    "Officials in {place} confirmed that the {topic} initiative received {num} in funding this year, the ministry reported.",
    "{person}, speaking at a press briefing in {place}, said the {topic} sector grew by {num} compared with last year.",
    "According to data released by {org}, {topic}-related activity increased {num} in the {year} fiscal year.",
    "{org} said in a statement that it had signed an agreement with {place} authorities to improve {topic} outcomes.",
    "The {topic} committee, chaired by {person}, will present its findings to parliament next month, a spokesperson said.",
    "Researchers at {org} published a peer-reviewed study showing a {num} change in {topic} trends since {year}.",
    "{person} told reporters in {place} that the government plans to invest {num} in {topic} infrastructure over the next {year}.",
    "A joint statement from {org} and local authorities in {place} confirmed the {topic} figures for {year}.",
    "The {topic} report, released by {org} on {day}, cited {num} as the key benchmark for this year's progress.",
    "{person} clarified during a {place} press conference that the {topic} data had been verified by independent auditors.",
    "Local authorities in {place} said {topic} services would expand by {num} following a review by {org}.",
    "Data from {org} indicates the {topic} sector added {num} jobs in {place} during {year}, officials confirmed.",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# ---------------------------------------------------------------------------
# FAKE NEWS TEMPLATES (sensational, unattributed, clickbait tone)
# ---------------------------------------------------------------------------
FAKE_TEMPLATES = [
    "SHOCKING: {person} secretly BANNED {topic} reforms and mainstream media refuses to report it!!!",
    "You won't believe what {org} is HIDING about {topic} — insiders reveal the TRUTH!",
    "BREAKING: Anonymous sources claim {person} covered up a {topic} scandal in {place}, and nobody is talking about it!",
    "This ONE {topic} secret from {place} will change your life forever, doctors HATE it!",
    "Leaked documents 'prove' {org} is lying about {topic} — share before this gets DELETED!",
    "Experts are STUNNED after {person} allegedly rigged the {topic} numbers in {place}!!!",
    "The government does NOT want you to know this shocking {topic} secret from {year}!",
    "Insider whistleblower says {org} faked {topic} data to control the public — wake up!!!",
    "You'll never guess what {person} did to {topic} funding — it's more evil than you think!",
    "URGENT: {topic} conspiracy in {place} exposed by 'anonymous insider', officials silent!!!",
    "Miracle {topic} trick banned by {org} because it threatens their profits, sources say!",
    "{person} caught on secret tape admitting the {topic} crisis was staged — media blackout!!!",
    "Is {org} hiding the real {topic} numbers? Anonymous sources say YES, and it's terrifying!",
    "SHARE THIS NOW before it's banned: the {topic} scandal {place} doesn't want revealed!",
    "They don't want you to see this: {person}'s secret {topic} plan that will SHOCK you!",
]

def fill(t):
    return t.format(
        person=random.choice(PEOPLE),
        org=random.choice(ORGS),
        place=random.choice(PLACES),
        topic=random.choice(TOPICS),
        num=random.choice(NUMBERS),
        year=random.choice(YEARS),
        day=random.choice(DAYS),
    )

def build_dataset(n_per_class=650):
    rows = []
    for _ in range(n_per_class):
        rows.append((fill(random.choice(REAL_TEMPLATES)), 1))
    for _ in range(n_per_class):
        rows.append((fill(random.choice(FAKE_TEMPLATES)), 0))
    random.shuffle(rows)
    # de-duplicate exact repeats while keeping class balance reasonably intact
    seen = set()
    unique_rows = []
    for text, label in rows:
        if text not in seen:
            seen.add(text)
            unique_rows.append((text, label))
    return unique_rows

def main():
    rows = build_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "news_dataset.csv")
    out_path = os.path.abspath(out_path)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    real_count = sum(1 for _, l in rows if l == 1)
    fake_count = sum(1 for _, l in rows if l == 0)
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"REAL: {real_count}  FAKE: {fake_count}")

if __name__ == "__main__":
    main()
