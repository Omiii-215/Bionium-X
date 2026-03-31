---
description: How to keep the Streamlit Cloud app alive and prevent it from sleeping
---

# Keep Streamlit App Alive

Streamlit Community Cloud puts apps to sleep after **12 hours of inactivity**.
A GitHub Actions cron job pings the app every **6 hours** to prevent this.

## Workflow File

`.github/workflows/keep_alive.yml`

## How It Works

1. GitHub Actions runs the `keep_alive.yml` workflow on a cron schedule (`0 */6 * * *`)
2. The job sends an HTTP GET request to `https://bionium-x.streamlit.app/`
3. This simulates a visitor, resetting the 12-hour inactivity timer
4. If the app doesn't respond with HTTP 200, the workflow logs a warning

## Manual Trigger

// turbo
1. Run `gh workflow run "Keep Streamlit App Alive"` from the repo root, or go to the GitHub **Actions** tab → **Keep Streamlit App Alive** → **Run workflow**

## Adjusting the Schedule

Edit the cron expression in `.github/workflows/keep_alive.yml`:

| Schedule        | Cron Expression  |
|-----------------|------------------|
| Every 6 hours   | `0 */6 * * *`    |
| Every 4 hours   | `0 */4 * * *`    |
| Every 8 hours   | `0 */8 * * *`    |
| Every 12 hours  | `0 */12 * * *`   |

> **Note:** Keep the interval strictly below 12 hours to stay within the inactivity window.

## Monitoring

- Go to the repo's **Actions** tab on GitHub
- Look for the **Keep Streamlit App Alive** workflow
- Green checkmarks = app was pinged successfully
- If you see failures, the app URL may have changed or the app may be deleted
