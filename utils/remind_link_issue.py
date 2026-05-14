# Sandbox copy of the PR-link-issue reminder bot.
# Differences from the upstream `huggingface/diffusers` version:
#   - REPO points at this sandbox repo.
#   - REMINDER_INTERVAL is 5 minutes (vs 7 days) for fast end-to-end tests.
"""
Periodically scan open PRs for a linked closing issue.

Behavior:
- Scans open, non-draft PRs.
- A PR is considered "linked" if GitHub's GraphQL `closingIssuesReferences` returns > 0
  (covers both `Fixes #N` keywords in the body and issues linked via the GitHub UI).
- If a PR is not linked, posts up to 3 reminder comments spaced REMINDER_INTERVAL apart.
- If the 3rd reminder is older than REMINDER_INTERVAL and the PR is still not linked, closes the PR.
- PRs labeled `no-issue-needed` and bot-authored PRs are skipped.
"""

import os
from datetime import datetime, timedelta, timezone

import requests
from github import Github


REPO = "sayakpaul/qwenimage-ptxla"
REMINDER_MARKER = "<!-- pr-link-issue-reminder -->"
CLOSE_MARKER = "<!-- pr-link-issue-close -->"
REMINDER_INTERVAL = timedelta(minutes=5)
MAX_REMINDERS = 3
BYPASS_LABELS = {"no-issue-needed"}

GRAPHQL_URL = "https://api.github.com/graphql"
GRAPHQL_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      closingIssuesReferences(first: 1) {
        totalCount
      }
    }
  }
}
"""


def _interval_label():
    seconds = int(REMINDER_INTERVAL.total_seconds())
    if seconds % 86400 == 0:
        n = seconds // 86400
        return f"{n} day" + ("s" if n != 1 else "")
    if seconds % 3600 == 0:
        n = seconds // 3600
        return f"{n} hour" + ("s" if n != 1 else "")
    if seconds % 60 == 0:
        n = seconds // 60
        return f"{n} minute" + ("s" if n != 1 else "")
    return f"{seconds} seconds"


def has_linked_issue(token, owner, name, number):
    response = requests.post(
        GRAPHQL_URL,
        json={"query": GRAPHQL_QUERY, "variables": {"owner": owner, "name": name, "number": number}},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["data"]["repository"]["pullRequest"]["closingIssuesReferences"]["totalCount"] > 0


def reminder_history(pr):
    reminders = [c for c in pr.get_issue_comments() if REMINDER_MARKER in (c.body or "")]
    reminders.sort(key=lambda c: c.created_at)
    return reminders


def reminder_body(author, count):
    remaining = MAX_REMINDERS - count
    window = _interval_label()
    lines = [
        REMINDER_MARKER,
        f"Hi @{author}, this PR does not appear to link an issue it fixes. "
        "If this PR addresses an existing issue, please add a closing keyword "
        "(e.g. `Fixes #1234`) to the PR description so the issue is linked.",
        "",
        f"Reminder **{count}/{MAX_REMINDERS}**. ",
    ]
    if remaining > 0:
        lines[-1] += (
            f"If no linked issue is added within {window}, "
            f"you will receive {remaining} more reminder(s)."
        )
    else:
        lines[-1] += (
            f"This is the final reminder. If no linked issue is added within {window}, "
            "this PR will be closed automatically. "
            "If this PR intentionally does not fix a tracked issue, a maintainer "
            "can add the `no-issue-needed` label to bypass this check."
        )
    return "\n".join(lines)


def close_body(author):
    return (
        f"{CLOSE_MARKER}\n"
        f"Closing this PR because @{author} did not add a linked issue after "
        f"{MAX_REMINDERS} reminders spaced {_interval_label()} apart. "
        "Please reopen once the PR description references the issue it fixes "
        "(e.g. `Fixes #1234`), or ask a maintainer to add the `no-issue-needed` "
        "label if this PR is intentionally unrelated to a tracked issue."
    )


def aware(ts):
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def main():
    token = os.environ["GITHUB_TOKEN"]
    g = Github(token)
    repo = g.get_repo(REPO)
    owner, name = REPO.split("/", 1)

    now = datetime.now(timezone.utc)

    for pr in repo.get_pulls(state="open"):
        if pr.draft:
            continue
        if pr.user is None:
            continue
        author = pr.user.login
        if not author or author.endswith("[bot]") or pr.user.type == "Bot":
            continue
        labels = {label.name for label in pr.labels}
        if labels & BYPASS_LABELS:
            continue
        if has_linked_issue(token, owner, name, pr.number):
            continue

        reminders = reminder_history(pr)
        count = len(reminders)

        if count == 0:
            pr.create_issue_comment(reminder_body(author, 1))
            continue

        if now - aware(reminders[-1].created_at) < REMINDER_INTERVAL:
            continue

        if count >= MAX_REMINDERS:
            pr.create_issue_comment(close_body(author))
            pr.edit(state="closed")
        else:
            pr.create_issue_comment(reminder_body(author, count + 1))


if __name__ == "__main__":
    main()
