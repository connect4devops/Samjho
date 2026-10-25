// Drives the real UI in a browser and captures the demo screenshots.
// Real inference against the local server - no mocking, no fabricated output.
const { chromium } = require('playwright');
const path = require('path');

const BASE = 'http://127.0.0.1:8090';
const IMG = path.resolve(__dirname, 'samples', 'whiteboard.png');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 940 } });

  page.on('console', m => {
    if (m.type() === 'error') console.log('  [console error]', m.text().slice(0, 160));
  });
  page.on('pageerror', e => console.log('  [page error]', String(e).slice(0, 160)));

  await page.goto(BASE, { waitUntil: 'networkidle' });
  await page.screenshot({ path: 'screenshots/01-landing.png' });
  console.log('captured landing');

  await page.setInputFiles('#file', IMG);
  await page.waitForSelector('#preview:visible', { timeout: 10000 });
  await page.screenshot({ path: 'screenshots/02-uploaded.png' });
  console.log('captured uploaded preview');

  await page.click('#run');
  await page.waitForSelector('.loading', { timeout: 5000 }).catch(() => {});
  await page.waitForTimeout(700);
  await page.screenshot({ path: 'screenshots/03-analyzing.png' });
  console.log('captured loading state');

  await page.waitForSelector('.tabs', { timeout: 240000 });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: 'screenshots/04-notes.png', fullPage: true });
  console.log('captured notes');

  for (const tab of ['takeaways', 'actions', 'uncertain', 'flow']) {
    const btn = page.locator(`.tab[data-tab="${tab}"]`);
    if (await btn.count() && await btn.isEnabled()) {
      await btn.click();
      await page.waitForTimeout(1400);
      await page.screenshot({ path: `screenshots/05-${tab}.png`, fullPage: true });
      console.log('captured', tab);
    } else {
      console.log('skipped (disabled/empty):', tab);
    }
  }

  const title = await page.locator('.title').textContent().catch(() => null);
  const sub = await page.locator('.sub').textContent().catch(() => null);
  console.log('\nRESULT TITLE:', (title || '').trim());
  console.log('RESULT SUB:', (sub || '').trim());

  await browser.close();
})().catch(e => { console.error('FAILED:', e.message); process.exit(1); });
