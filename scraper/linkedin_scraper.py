Discarded...

from playwright.sync_api import sync_playwright

def scrape_google_jobs(keywords="software engineer", location="New York, USA"):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        query = f"{keywords} jobs in {location}"
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}&ibp=htl;jobs"
        page.goto(url, wait_until="networkidle")

        # Wait for the jobs panel container to load with a reasonable timeout
        page.wait_for_timeout(30000)

        # Locate the job posting divs by jsname attribute
        job_postings = page.locator('div[jsname="cKdk8"]')
        count = job_postings.count()
        print(f"Found {count} job postings")

        jobs = []
        for i in range(min(count, 10)):
            job = job_postings.nth(i)
            try:
                title = job.locator('div[role="heading"]').first.inner_text(timeout=1000).strip()
            except:
                title = ""
            try:
                # Try alternative company selector by narrowing spans
                company = job.locator('div:has(span)').nth(0).inner_text(timeout=1000).strip()
            except:
                company = ""
            try:
                # Location often nearby or in sibling div/span with a specific class
                location = job.locator('span[jsname="vWLAgc"]').first.inner_text(timeout=1000).strip()
            except:
                location = ""
            try:
                job.click()
                page.wait_for_timeout(1500)
                description = page.locator('div[jsname="Wct42"]').inner_text(timeout=1000).strip()
            except:
                description = ""

            jobs.append({
                "title": title,
                "company": company,
                "location": location,
                "description": description
            })

        browser.close()
        return jobs

if __name__ == "__main__":
    jobs = scrape_google_jobs()
    for idx, job in enumerate(jobs, 1):
        print(f"Job #{idx}:")
        for k, v in job.items():
            print(f"{k.capitalize()}: {v[:100]}")
        print("-" * 40)
